# Building

```bash
mkdir build
cd build

cmake ..

make -j
```

The hipcc and hipconfig executables are created in the current build folder. These executables need to be copied to /opt/rocm/hip/bin folder location. Packaging and installing will be handled in future releases.
