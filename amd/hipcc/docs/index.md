# HIPCC

`hipcc` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure. Historically, `hipcc` was provided as a script in the HIP repo ( https://github.com/ROCm-Developer-Tools/HIP/blob/main/bin/hipcc ). The `hipcc` provided in this project provides the same functionality, but is a binary rather than a script. At some point in the future, the hipcc script will be deprecated and ultimately removed from the HIP repo.

`hipcc` will pass-through options to the target compiler. The tools calling hipcc must ensure the compiler options are appropriate for the target compiler.
