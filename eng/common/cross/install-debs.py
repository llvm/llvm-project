#!/usr/bin/env python3

import argparse
import asyncio
import aiohttp
import gzip
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zstandard

from collections import deque
from functools import cmp_to_key

async def download_file(session, url, dest_path, max_retries=3, retry_delay=2, timeout=60):
    """Asynchronous file download with retries."""
    attempt = 0
    while attempt < max_retries:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    with open(dest_path, "wb") as f:
                        content = await response.read()
                        f.write(content)
                    print(f"Downloaded {url} at {dest_path}")
                    return
                else:
                    print(f"Failed to download {url}, Status Code: {response.status}")
                    break
        except (asyncio.CancelledError, asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"Error downloading {url}: {type(e).__name__} - {e}. Retrying...")

        attempt += 1
        await asyncio.sleep(retry_delay)

    print(f"Failed to download {url} after {max_retries} attempts.")

async def download_deb_files_parallel(mirror, packages, tmp_dir):
    """Download .deb files in parallel."""
    os.makedirs(tmp_dir, exist_ok=True)

    tasks = []
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for pkg, info in packages.items():
            filename = info.get("Filename")
            if filename:
                url = f"{mirror}/{filename}"
                dest_path = os.path.join(tmp_dir, os.path.basename(filename))
                tasks.append(asyncio.create_task(download_file(session, url, dest_path)))

        await asyncio.gather(*tasks)

async def download_package_index_parallel(mirror, arch, suites):
    """Download package index files for specified suites and components entirely in memory."""
    tasks = []
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for suite in suites:
            for component in ["main", "universe"]:
                url = f"{mirror}/dists/{suite}/{component}/binary-{arch}/Packages.gz"
                tasks.append(fetch_and_decompress(session, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    merged_content = ""
    for result in results:
        if isinstance(result, str):
            if merged_content:
                merged_content += "\n\n"
            merged_content += result

    return merged_content

async def fetch_and_decompress(session, url):
    """Fetch and decompress the Packages.gz file."""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                compressed_data = await response.read()
                decompressed_data = gzip.decompress(compressed_data).decode('utf-8')
                print(f"Downloaded index: {url}")
                return decompressed_data
            else:
                print(f"Skipped index: {url} (doesn't exist)")
                return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")

def parse_debian_version(version):
    """Parse a Debian package version into epoch, upstream version, and revision."""
    match = re.match(r'^(?:(\d+):)?([^-]+)(?:-(.+))?$', version)
    if not match:
        raise ValueError(f"Invalid Debian version format: {version}")
    epoch, upstream, revision = match.groups()
    return int(epoch) if epoch else 0, upstream, revision or ""

def compare_upstream_version(v1, v2):
    """Compare upstream or revision parts using Debian rules."""
    def tokenize(version):
        tokens = re.split(r'([0-9]+|[A-Za-z]+)', version)
        return [int(x) if x.isdigit() else x for x in tokens if x]

    tokens1 = tokenize(v1)
    tokens2 = tokenize(v2)

    for token1, token2 in zip(tokens1, tokens2):
        if type(token1) == type(token2):
            if token1 != token2:
                return (token1 > token2) - (token1 < token2)
        else:
            return -1 if isinstance(token1, str) else 1

    return len(tokens1) - len(tokens2)

def compare_debian_versions(version1, version2):
    """Compare two Debian package versions."""
    epoch1, upstream1, revision1 = parse_debian_version(version1)
    epoch2, upstream2, revision2 = parse_debian_version(version2)

    if epoch1 != epoch2:
        return epoch1 - epoch2

    result = compare_upstream_version(upstream1, upstream2)
    if result != 0:
        return result

    return compare_upstream_version(revision1, revision2)

def resolve_dependencies(packages, aliases, desired_packages):
    """Recursively resolves dependencies for the desired packages."""
    resolved = []
    to_process = deque(desired_packages)

    while to_process:
        current = to_process.popleft()
        resolved_package = current if current in packages else aliases.get(current, [None])[0]

        if not resolved_package:
            print(f"Error: Package '{current}' was not found in the available packages.")
            sys.exit(1)

        if resolved_package not in resolved:
            resolved.append(resolved_package)

            deps = packages.get(resolved_package, {}).get("Depends", "")
            if deps:
                deps = [dep.split(' ')[0] for dep in deps.split(', ') if dep]
                for dep in deps:
                    if dep not in resolved and dep not in to_process and dep in packages:
                        to_process.append(dep)

    return resolved

def parse_package_index(content):
    """Parses the Packages.gz file and returns package information."""
    packages = {}
    aliases = {}
    entries = re.split(r'\n\n+', content)

    for entry in entries:
        fields = dict(re.findall(r'^(\S+): (.+)$', entry, re.MULTILINE))
        if "Package" in fields:
            package_name = fields["Package"]
            version = fields.get("Version")
            filename = fields.get("Filename")
            depends = fields.get("Depends")
            provides = fields.get("Provides", None)

            # Only update if package_name is not in packages or if the new version is higher
            if package_name not in packages or compare_debian_versions(version, packages[package_name]["Version"]) > 0:
                packages[package_name] = {
                    "Version": version,
                    "Filename": filename,
                    "Depends": depends
                }

                # Update aliases if package provides any alternatives
                if provides:
                    provides_list = [x.strip() for x in provides.split(",")]
                    for alias in provides_list:
                        # Strip version specifiers
                        alias_name = re.sub(r'\s*\(=.*\)', '', alias)
                        if alias_name not in aliases:
                            aliases[alias_name] = []
                        if package_name not in aliases[alias_name]:
                            aliases[alias_name].append(package_name)

    return packages, aliases

def install_packages(mirror, packages_info, aliases, tmp_dir, extract_dir, ar_tool, desired_packages):
    """Downloads .deb files and extracts them."""
    resolved_packages = resolve_dependencies(packages_info, aliases, desired_packages)
    print(f"Resolved packages (including dependencies): {resolved_packages}")

    packages_to_download = {}

    for pkg in resolved_packages:
        if pkg in packages_info:
            packages_to_download[pkg] = packages_info[pkg]

        if pkg in aliases:
            for alias in aliases[pkg]:
                if alias in packages_info:
                    packages_to_download[alias] = packages_info[alias]

    asyncio.run(download_deb_files_parallel(mirror, packages_to_download, tmp_dir))

    package_to_deb_file_map = {}
    for pkg in resolved_packages:
        pkg_info = packages_info.get(pkg)
        if pkg_info:
            deb_filename = pkg_info.get("Filename")
            if deb_filename:
                deb_file_path = os.path.join(tmp_dir, os.path.basename(deb_filename))
                package_to_deb_file_map[pkg] = deb_file_path

    for pkg in reversed(resolved_packages):
        deb_file = package_to_deb_file_map.get(pkg)
        if deb_file and os.path.exists(deb_file):
            extract_deb_file(deb_file, tmp_dir, extract_dir, ar_tool)

    print("All done!")

def extract_deb_file(deb_file, tmp_dir, extract_dir, ar_tool):
    """Extract .deb file contents"""

    os.makedirs(extract_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_subdir:
        result = subprocess.run(f"{ar_tool} t {os.path.abspath(deb_file)}", cwd=tmp_subdir, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        tar_filename = None
        for line in result.stdout.decode().splitlines():
            if line.startswith("data.tar"):
                tar_filename = line.strip()
                break

        if not tar_filename:
            raise FileNotFoundError(f"Could not find 'data.tar.*' in {deb_file}.")

        tar_file_path = os.path.join(tmp_subdir, tar_filename)
        print(f"Extracting {tar_filename} from {deb_file}..")

        subprocess.run(f"{ar_tool} p {os.path.abspath(deb_file)} {tar_filename} > {tar_file_path}", check=True, shell=True)

        file_extension = os.path.splitext(tar_file_path)[1].lower()

        if file_extension == ".xz":
            mode = "r:xz"
        elif file_extension == ".gz":
            mode = "r:gz"
        elif file_extension == ".zst":
            # zstd is not supported by standard library yet
            decompressed_tar_path = tar_file_path.replace(".zst", "")
            with open(tar_file_path, "rb") as zst_file, open(decompressed_tar_path, "wb") as decompressed_file:
                dctx = zstandard.ZstdDecompressor()
                dctx.copy_stream(zst_file, decompressed_file)

            tar_file_path = decompressed_tar_path
            mode = "r"
        else:
            raise ValueError(f"Unsupported compression format: {file_extension}")

        with tarfile.open(tar_file_path, mode) as tar:
            tar.extractall(path=extract_dir, filter='fully_trusted')

def finalize_setup(rootfsdir):
    lib_dir = os.path.join(rootfsdir, 'lib')
    usr_lib_dir = os.path.join(rootfsdir, 'usr', 'lib')

    if os.path.exists(lib_dir):
        if os.path.islink(lib_dir):
            os.remove(lib_dir)
        else:
            os.makedirs(usr_lib_dir, exist_ok=True)

            for item in os.listdir(lib_dir):
                src = os.path.join(lib_dir, item)
                dest = os.path.join(usr_lib_dir, item)

                if os.path.isdir(src):
                    shutil.copytree(src, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dest)

            shutil.rmtree(lib_dir)

    os.symlink(usr_lib_dir, lib_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rootfs for .NET runtime on Debian-like OS")
    parser.add_argument("--distro", required=False, help="Distro name (e.g., debian, ubuntu, etc.)")
    parser.add_argument("--arch", required=True, help="Architecture (e.g., amd64, loong64, etc.)")
    parser.add_argument("--rootfsdir", required=True, help="Destination directory.")
    parser.add_argument('--suite', required=True, action='append', help='Specify one or more repository suites to collect index data.')
    parser.add_argument("--mirror", required=False, help="Mirror (e.g., http://ftp.debian.org/debian-ports etc.)")
    parser.add_argument("--artool", required=False, default="ar", help="ar tool to extract debs (e.g., ar, llvm-ar etc.)")
    parser.add_argument("packages", nargs="+", help="List of package names to be installed.")

    args = parser.parse_args()

    if args.mirror is None:
        if args.distro == "ubuntu":
            args.mirror = "http://archive.ubuntu.com/ubuntu" if args.arch in ["amd64", "i386"] else "http://ports.ubuntu.com/ubuntu-ports"
        elif args.distro == "debian":
            args.mirror = "http://ftp.debian.org/debian-ports"
        else:
            raise Exception("Unsupported distro")

    DESIRED_PACKAGES = args.packages + [ # base packages
        "dpkg",
        "busybox",
        "libc-bin",
        "base-files",
        "base-passwd",
        "debianutils"
    ]

    print(f"Creating rootfs. rootfsdir: {args.rootfsdir}, distro: {args.distro}, arch: {args.arch}, suites: {args.suite}, mirror: {args.mirror}")

    package_index_content = asyncio.run(download_package_index_parallel(args.mirror, args.arch, args.suite))

    packages_info, aliases = parse_package_index(package_index_content)

    with tempfile.TemporaryDirectory() as tmp_dir:
        install_packages(args.mirror, packages_info, aliases, tmp_dir, args.rootfsdir, args.artool, DESIRED_PACKAGES)

    finalize_setup(args.rootfsdir)
