import subprocess


def getRoot(config):
    if not config.parent:
        return config
    return getRoot(config.parent)


root = getRoot(config)

if root.target_os not in ["Darwin"]:
    config.unsupported = True


def get_product_version():
    try:
        version_process = subprocess.run(
            ["sw_vers", "-productVersion"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        version_string = version_process.stdout.decode("utf-8").split("\n")[0]
        version_split = version_string.split(".")
        return (int(version_split[0]), int(version_split[1]))
    except:
        return (0, 0)


macos_version_major, macos_version_minor = get_product_version()
if macos_version_major > 10 and macos_version_minor > 11:
    config.available_features.add("mac-os-10-11-or-higher")
else:
    config.available_features.add("mac-os-10-10-or-lower")
