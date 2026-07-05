.. title:: clang-tidy - altera-struct-pack-align

altera-struct-pack-align
========================

Finds structs that are inefficiently packed or aligned, and recommends
packing and/or aligning of said structs as needed.

Structs that are not packed take up more space than they should, and accessing
structs that are not well aligned is inefficient.

Fix-its are provided to fix both of these issues by inserting and/or amending
relevant struct attributes.

Based on the `Altera SDK for OpenCL: Best Practices Guide
<https://www.altera.com/en_US/pdfs/literature/hb/opencl-sdk/aocl_optimization_guide.pdf>`_.

.. code-block:: c++

  // The following struct is originally aligned to 4 bytes, and thus takes up
  // 12 bytes of memory instead of 10. Packing the struct will make it use
  // only 10 bytes of memory, and aligning it to 16 bytes will make it
  // efficient to access.
  struct example {
    char a;    // 1 byte
    double b;  // 8 bytes
    char c;    // 1 byte
  };

  // The following struct is arranged in such a way that packing is not needed.
  // However, it is aligned to 4 bytes instead of 8, and thus needs to be
  // explicitly aligned.
  struct implicitly_packed_example {
    char a;  // 1 byte
    char b;  // 1 byte
    char c;  // 1 byte
    char d;  // 1 byte
    int e;   // 4 bytes
  };

  // The following struct is explicitly aligned and packed.
  struct good_example {
    char a;    // 1 byte
    double b;  // 8 bytes
    char c;    // 1 byte
  } __attribute__((packed)) __attribute__((aligned(16));

  // Explicitly aligning a struct to the wrong value will result in a warning.
  // The following example should be aligned to 16 bytes, not 32.
  struct badly_aligned_example {
    char a;    // 1 byte
    double b;  // 8 bytes
    char c;    // 1 byte
  } __attribute__((packed)) __attribute__((aligned(32)));
