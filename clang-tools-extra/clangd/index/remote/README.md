# Clangd remote index

Clangd uses a global index for project-wide code completion, navigation and
other features.  For large projects, building this can take many hours and
keeping it loaded uses a lot of memory.

To relieve that burden, we're building remote index &mdash; a global index
served on a different machine and shared between developers. This directory
contains code that is used as Proof of Concept for the upcoming remote index
feature.

## Building

This feature uses gRPC and Protobuf libraries, so you will need to install them.
There are two ways of doing that.

However you install dependencies, to enable this feature and build remote index
tools you will need to set this CMake flag &mdash; `-DCLANGD_ENABLE_REMOTE=On`.

### System-installed libraries

On Debian-like systems gRPC and Protobuf can be installed from apt:

```bash
apt install libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
```

### Building from sources

Another way of installing gRPC and Protobuf is building from sources using
CMake (we need CMake config files to find necessary libraries in LLVM). The
easiest way of doing that would be to choose a directory where you want to
install so that the installation files are not copied to system root and you
can easily uninstall gRPC or use different versions.

```bash
# Get source code.
$ git clone -b v1.36.3 https://github.com/grpc/grpc
$ cd grpc
$ git submodule update --init
# Choose directory where you want gRPC installation to live.
$ export GRPC_INSTALL_PATH=/where/you/want/grpc/to/be/installed
# Build and install gRPC to ${GRPC_INSTALL_PATH}
$ mkdir build; cd build
$ cmake -DgRPC_INSTALL=ON -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_PATH} -DCMAKE_BUILD_TYPE=Release ..
$ make install
```

This [guide](https://github.com/grpc/grpc/blob/master/BUILDING.md) goes into
more detail on how to build gRPC from sources.

By default, CMake will look for system-installed libraries when building remote
index tools so you will have to adjust LLVM's CMake invocation. The following
flag will inform build system that you chose this option &mdash;
`-DGRPC_INSTALL_PATH=${GRPC_INSTALL_PATH}`.

## Running

You can run `clangd-index-server` and connect `clangd` instance to it using
`--remote-index-address` and `--project-root` flags.
