# Agent image for LLVM org cluster.
# .net 4.8 is required by chocolately package manager.
FROM mcr.microsoft.com/dotnet/framework/sdk:4.8-windowsservercore-ltsc2019

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
RUN choco install -y ninja git
# Pin an older version of Python; the current Python 3.10 fails when
# doing "pip install" for the other dependencies, as it fails to find libxml
# while compiling some package.
RUN choco install -y python3 --version 3.9.7

# ActivePerl is currently not installable via Chocolatey, see
# http://disq.us/p/2ipditb. Install StrawberryPerl instead. Unfortunately,
# StrawberryPerl not only installs Perl, but also a redundant C/C++ compiler
# toolchain, and a copy of pkg-config which can cause misdetections for other
# built products, see
# https://github.com/StrawberryPerl/Perl-Dist-Strawberry/issues/11 for further
# details. Remove the redundant and unnecessary parts of the StrawberryPerl
# install.
RUN choco install -y strawberryperl && \
    rmdir /q /s c:\strawberry\c && \
    del /q c:\strawberry\perl\bin\pkg-config*

# libcxx requires clang(-cl) to be available
RUN choco install -y sccache llvm
RUN pip install psutil

RUN curl -LO https://github.com/mstorsjo/llvm-mingw/releases/download/20230320/llvm-mingw-20230320-ucrt-x86_64.zip && \
    powershell Expand-Archive llvm-mingw-*-ucrt-x86_64.zip -DestinationPath . && \
    del llvm-mingw-*-ucrt-x86_64.zip && \
    ren llvm-mingw-20230320-ucrt-x86_64 llvm-mingw

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

# handle for debugging of files beeing locked by some processes.
RUN choco install -y handle

RUN pip3 install pywin32 buildbot-worker==2.8.4

ARG RUNNER_VERSION=2.319.1
ENV RUNNER_VERSION=$RUNNER_VERSION

RUN powershell -Command \
    Invoke-WebRequest -Uri https://github.com/actions/runner/releases/download/v${env:RUNNER_VERSION}/actions-runner-win-x64-${env:RUNNER_VERSION}.zip -OutFile actions-runner-win.zip ; \
    Add-Type -AssemblyName System.IO.Compression.FileSystem ; \
    [System.IO.Compression.ZipFile]::ExtractToDirectory('actions-runner-win.zip', $PWD) ;\
    rm actions-runner-win.zip
