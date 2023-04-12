# HIP compiler driver (hipcc)

## Table of Contents

<!-- toc -->

- [hipcc](#hipcc)
     * [Documentation](#documentation)
     * [Environment Variables](#envVar)
     * [Usage](#hipcc-usage)
     * [Building](#building)
     * [Testing](#testing)

<!-- tocstop -->

## <a name="hipcc"></a> hipcc

`hipcc` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure. Historically, `hipcc` was provided as a script in the HIP repo ( https://github.com/ROCm-Developer-Tools/HIP/blob/main/bin/hipcc ). The `hipcc` provided in this project provides the same functionality, but is a binary rather than a script. At some point in the future, the hipcc script will be deprecated and ultimately removed from the HIP repo.

`hipcc` will pass-through options to the target compiler. The tools calling hipcc must ensure the compiler options are appropriate for the target compiler.

## Documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r .sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

### <a name="envVar"></a> Environment Variables

The environment variable HIP_PLATFORM may be used to specify amd/nvidia:
- HIP_PLATFORM='amd' or HIP_PLATFORM='nvidia'.
- If HIP_PLATFORM is not set, then hipcc will attempt to auto-detect based on if nvcc is found.

Other environment variable controls:
- HIP_PATH        : Path to HIP directory, default is one dir level above location of hipcc.
- CUDA_PATH       : Path to CUDA SDK (default /usr/local/cuda). Used on NVIDIA platforms only.
- HSA_PATH        : Path to HSA dir (defaults to ../../hsa relative to abs_path of hipcc). Used on AMD platforms only.
- HIP_ROCCLR_HOME : Path to HIP/ROCclr directory. Used on AMD platforms only.
- HIP_CLANG_PATH  : Path to HIP-Clang (default to ../../llvm/bin relative to hipcc's abs_path). Used on AMD platforms only.

### <a name="usage"></a> hipcc: usage

The built executables can be used the same way as the hipcc/hipconfig perl scripts. 
To use the newly built executables from the build folder use ./ in front of the executable name -
Example:
```shell
./hipconfig --help
./hipcc --help
./hipcc --version
./hipconfig --full
```

when the excutables are copied to /opt/rocm/hip/bin or <anyfolder>hip/bin. 
The ./ is not required as the HIP path is added to the envirnoment variables list.

### <a name="building"></a> hipcc: building

```bash
mkdir build
cd build

cmake ..

make -j
```

The hipcc and hipconfig executables are created in the current build folder. These executables need to be copied to /opt/rocm/hip/bin folder location. Packaging and installing will be handled in future releases.

### <a name="testing"></a> hipcc: testing

Currently hipcc/hipconfig executables are tested by building and executing HIP tests. Seperate tests for hipcc/hipconfig is currently not planned.   
