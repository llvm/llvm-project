# Agent image for LLVM org cluster.
# .net 4.8 is required by chocolately package manager.
FROM mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2022

# Restore the default Windows shell for correct batch processing.
SHELL ["cmd", "/S", "/C"]

# Download the Build Tools bootstrapper.
ADD https://aka.ms/vs/16/release/vs_buildtools.exe /TEMP/vs_buildtools.exe

RUN powershell -Command Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Download channel for fixed install.
ARG CHANNEL_URL=https://aka.ms/vs/16/release/channel
ADD ${CHANNEL_URL} /TEMP/VisualStudio.chman

# Install Build Tools with C++ workload.
#   - Documentation for docker installation
#     https://docs.microsoft.com/en-us/visualstudio/install/build-tools-container?view=vs-2019
#   - Documentation on workloads
#     https://docs.microsoft.com/en-us/visualstudio/install/workload-component-id-vs-build-tools?view=vs-2019#c-build-tools
#   - Documentation on flags
#     https://docs.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2019
RUN /TEMP/vs_buildtools.exe --quiet --wait --norestart --nocache \
    --channelUri C:\TEMP\VisualStudio.chman \
    --installChannelUri C:\TEMP\VisualStudio.chman \
    --installPath C:\BuildTools \
    --add Microsoft.VisualStudio.Workload.VCTools \
    --add Microsoft.VisualStudio.Component.VC.ATL \
    --includeRecommended \
    || IF "%ERRORLEVEL%"=="3010" EXIT 0

# Register DIA dll (Debug Interface Access) so it can be used to symbolize
# the stack traces. Register dll for 32 and 64 bit.
# see https://developercommunity.visualstudio.com/content/problem/290674/msdia140dll-is-not-registered-on-vs2017-hosts.html

RUN regsvr32 /S "C:\BuildTools\DIA SDK\bin\amd64\msdia140.dll" & \
    regsvr32 /S "C:\BuildTools\DIA SDK\bin\msdia140.dll"

# install tools as described in https://llvm.org/docs/GettingStartedVS.html
# and a few more that were not documented...
# Pin an older version of Python; the current Python 3.10 fails when
# doing "pip install" for the other dependencies, as it fails to find libxml
# while compiling some package.
# We version pin the other packages as well to ensure the container build is as
# reproducible as possible to prevent issues when upgrading only part of the
# container.
RUN choco install -y ninja --version 1.13.1 && \
    choco install -y git --version 2.50.1 && \
    choco install -y sccache --version 0.10.0 && \
    choco install -y python3 --version 3.9.7

# Testing requires psutil
RUN pip install psutil

# configure Python encoding
ENV PYTHONIOENCODING=UTF-8

# update the path variable
# C:\Program Files\Git\usr\bin contains a usable bash and other unix tools.
# C:\llvm-mingw\bin contains Clang configured for mingw targets and
#     corresponding sysroots. Both the 'llvm' package (with Clang defaulting
#     to MSVC targets) and this directory contains executables named
#     'clang.exe' - add this last to let the other one have precedence.
#     To use these compilers, use the triple prefixed form, e.g.
#     x86_64-w64-mingw32-clang.
# C:\buildtools and SDK paths are ones that are set by c:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=amd64 -host_arch=amd64
RUN powershell -Command \
    [System.Environment]::SetEnvironmentVariable('PATH', \
    [System.Environment]::GetEnvironmentVariable('PATH', 'machine') + ';C:\Program Files\Git\usr\bin;C:\llvm-mingw\bin' \
    + ';C:\BuildTools\Common7\IDE\' \
    + ';C:\BuildTools\Common7\IDE\CommonExt ensions\Microsoft\TeamFoundation\Team Explorer' \
    + ';C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin' \
    + ';C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja' \
    + ';C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer' \
    + ';C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\TestWindow' \
    + ';C:\BuildTools\Common7\IDE\VC\VCPackages' \
    + ';C:\BuildTools\Common7\Tools\' \
    + ';C:\BuildTools\Common7\Tools\devinit' \
    + ';C:\BuildTools\MSBuild\Current\Bin' \
    + ';C:\BuildTools\MSBuild\Current\bin\Roslyn' \
    + ';C:\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64' \
    + ';C:\Program Files (x86)\Microsoft SDKs\Windows\v10.0A\bin\NETFX 4.8 Tools\x64\' \
    + ';C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64' \
    + ';C:\Program Files (x86)\Windows Kits\10\bin\x64' \
    + ';C:\Windows\Microsoft.NET\Framework64\v4.0.30319' \
    ,'machine')

# support long file names during git checkout
RUN git config --system core.longpaths true & \
    git config --global core.autocrlf false

ARG RUNNER_VERSION=2.329.0
ENV RUNNER_VERSION=$RUNNER_VERSION

RUN powershell -Command \
    Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v${env:RUNNER_VERSION}/actions-runner-win-x64-${env:RUNNER_VERSION}.zip -OutFile actions-runner-win.zip ; \
    Add-Type -AssemblyName System.IO.Compression.FileSystem ; \
    [System.IO.Compression.ZipFile]::ExtractToDirectory('actions-runner-win.zip', $PWD) ;\
    rm actions-runner-win.zip
