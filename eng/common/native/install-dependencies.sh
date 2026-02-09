#!/bin/sh

set -e

# This is a simple script primarily used for CI to install necessary dependencies
#
# Usage:
#
# ./install-dependencies.sh <OS>

os="$(echo "$1" | tr "[:upper:]" "[:lower:]")"

if [ -z "$os" ]; then
    . "$(dirname "$0")"/init-os-and-arch.sh
fi

case "$os" in
    linux)
        if [ -e /etc/os-release ]; then
            . /etc/os-release
        fi

        if [ "$ID" = "debian" ] || [ "$ID_LIKE" = "debian" ]; then
            apt update

            apt install -y build-essential gettext locales cmake llvm clang lld lldb liblldb-dev libunwind8-dev libicu-dev liblttng-ust-dev \
                libssl-dev libkrb5-dev pigz cpio ninja-build

            localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
        elif [ "$ID" = "fedora" ] || [ "$ID" = "rhel" ] || [ "$ID" = "azurelinux" ] || [ "$ID" = "centos" ]; then
            pkg_mgr="$(command -v tdnf 2>/dev/null || command -v dnf)"
            $pkg_mgr install -y cmake llvm lld lldb clang python curl libicu-devel openssl-devel krb5-devel lttng-ust-devel pigz cpio ninja-build
        elif [ "$ID" = "amzn" ]; then
            dnf install -y cmake llvm lld lldb clang python libicu-devel openssl-devel krb5-devel lttng-ust-devel pigz cpio ninja-build
        elif [ "$ID" = "alpine" ]; then
            apk add build-base cmake bash curl clang llvm-dev lld lldb krb5-dev lttng-ust-dev icu-dev openssl-dev pigz cpio ninja
        else
            echo "Unsupported distro. distro: $ID"
            exit 1
        fi
        ;;

    osx|maccatalyst|ios|iossimulator|tvos|tvossimulator)
        echo "Installed xcode version: $(xcode-select -p)"

        export HOMEBREW_NO_INSTALL_CLEANUP=1
        export HOMEBREW_NO_INSTALLED_DEPENDENTS_CHECK=1
        # Skip brew update for now, see https://github.com/actions/setup-python/issues/577
        # brew update --preinstall
        brew bundle --no-upgrade --file=- <<EOF
brew "cmake"
brew "icu4c"
brew "openssl@3"
brew "pkgconf"
brew "python3"
brew "pigz"
brew "ninja"
EOF
        ;;

    *)
        echo "Unsupported platform. OS: $os"
        exit 1
        ;;
esac
