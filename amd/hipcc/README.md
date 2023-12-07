# HIP compiler driver (hipcc)

## Table of Contents

<!-- toc -->

- [hipcc](#hipcc)
  - [Documentation](#documentation)
  - [Environment Variables](#envVar)
  - [Usage](#usage)
  - [Building](#building)
  - [Testing](#testing)

<!-- tocstop -->

## <a name="hipcc"></a> hipcc

`hipcc` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure. 

`hipcc` will pass-through options to the target compiler. The tools calling hipcc must ensure the compiler options are appropriate for the target compiler.

### <a name="building"></a> Building

Building on Linux:

```bash
mkdir build
cd build

cmake ..

make -j4
```

The hipcc and hipconfig executables are created in the current build folder. 
You may also create installable packages with :
```bash
make package
```

## Documentation

Run the steps below to build documentation locally.

```shell
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

### <a name="envVar"></a> Environment Variables

The environment variable HIP_PLATFORM may be used to specify amd/nvidia:

- HIP_PLATFORM='amd' or HIP_PLATFORM='nvidia'.
- If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.

Other environment variable controls:

- CUDA_PATH       : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.

### <a name="usage"></a> Usage

The built executables can be used the same way as the hipcc/hipconfig perl scripts. 
To use the newly built executables from the build folder use ./ in front of the executable name -
Example:
```shell
./hipconfig --help
./hipcc --help
./hipcc --version
./hipconfig --full
```

### <a name="testing"></a> hipcc: testing

Currently hipcc/hipconfig executables are tested by building and executing HIP tests: https://github.com/ROCm/hip-tests
