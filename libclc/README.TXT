libclc
------

libclc is an open source, BSD licensed implementation of the library
requirements of the OpenCL C programming language, as specified by the
OpenCL 1.1 Specification. The following sections of the specification
impose library requirements:

  * 6.1: Supported Data Types
  * 6.2.3: Explicit Conversions
  * 6.2.4.2: Reinterpreting Types Using as_type() and as_typen()
  * 6.9: Preprocessor Directives and Macros
  * 6.11: Built-in Functions
  * 9.3: Double Precision Floating-Point
  * 9.4: 64-bit Atomics
  * 9.5: Writing to 3D image memory objects
  * 9.6: Half Precision Floating-Point

libclc is intended to be used with the Clang compiler's OpenCL frontend.

libclc is designed to be portable and extensible. To this end, it provides
generic implementations of most library requirements, allowing the target
to override the generic implementation at the granularity of individual
functions.

libclc currently only supports the PTX target, but support for more
targets is welcome.

Compiling and installing with Make
----------------------------------

$ ./configure.py --with-llvm-config=/path/to/llvm-config && make
$ make install

Note you can use the DESTDIR Makefile variable to do staged installs.

$ make install DESTDIR=/path/for/staged/install

Compiling and installing with Ninja
-----------------------------------

$ ./configure.py -g ninja --with-llvm-config=/path/to/llvm-config && ninja
$ ninja install

Note you can use the DESTDIR environment variable to do staged installs.

$ DESTDIR=/path/for/staged/install ninja install

Website
-------

https://libclc.llvm.org/


---
OpenCL类比与英伟达的CUDA，是第一个面向异构系统(此系统中可由CPU，GPU或其它类型的处理器架构组成)的并行编程的开放式标准。

OpenCL 是一个异构并行计算平台，用于编写可以在不同类型的计算设备上运行的程序。它支持将计算任务映射到 CPU、GPU、DSP 和 FPGA 等不同的硬件上。OpenCL 提供了一个底层硬件结构的抽象模型，并旨在提供一个通用的开发 API，使得开发人员可以编写在 GPU 上运行的通用计算程序，而无需将其算法映射到 OpenGL 或 DirectX 的 3D 图形 API 上。


相较于CUDA只支持英伟达的GPU，OpenCL更加通用，但也因为需要通用而导致去实现复杂性较高，目前CUDA更加流行。
