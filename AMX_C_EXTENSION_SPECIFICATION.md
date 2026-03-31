# AMX C Extension Specification

## Overview
This document outlines the comprehensive specifications for four major issues related to the AMX (Advanced Matrix Extensibility) extension in the LLVM project.

### 1. Tilei8 Fundamental Type Implementation
The Tilei8 fundamental type must support both 16x16 and 8x8 sizes. The following implementation details are provided:
#### Implementation Details
- **16x16 Size**: Define the Tilei8 type to represent a 16x16 matrix of i8 integers, including appropriate loading, storing, and manipulation operations.
- **8x8 Size**: Define the Tilei8 type for an 8x8 matrix, ensuring that operations are correctly optimized for this size as well.  
\n#### Reference Implementation: `X86LowerAMXType.cpp`
```cpp
// Example of Tilei8 type handling in LLVM
struct Tilei8 {
    int8_t data[16][16];
};

Tilei8 loadTilei8(const int8_t* src);
void storeTilei8(Tilei8 tile, int8_t* dst);
```

### 2. Tile Zero Initialization Mapping to _tile_zero Intrinsic
The mapping of zero initialization in tiles must be efficiently translated to the `_tile_zero` intrinsic.
#### Implementation Details
- Ensure that any zero-initialization in the AMX context can leverage the `_tile_zero` intrinsic to improve performance.

#### Reference Implementation: `X86LowerAMXIntrinsics.cpp`
```cpp
// Example intrinsic mapping
void tile_zero(Tilei8& tile) {
    asm("_tile_zero %0" : ":r"(tile));
}
```

### 3. Pointer Assignment Loading for Tile Loads
Pointer assignment should seamlessly integrate with tile loading operations.
#### Implementation Details
- Define how pointers can be assigned during tile load operations, ensuring consistent and optimized access patterns for the corresponding tile types.

#### Reference Implementation: `X86PreAMXConfig.cpp`
```cpp
// Example of pointer loading handling
Tilei8 loadTileFromPointer(int8_t* ptr) {
    Tilei8 tile;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            tile.data[i][j] = ptr[i * 16 + j];
        }
    }
    return tile;
}
```

### 4. AMX Extension Control Flag (-fintel-amx option)
The -fintel-amx option must be accurately processed to enable or disable the AMX extension features.
#### Implementation Details
- The compiler flags must handle the -fintel-amx option appropriately, controlling the activation of the AMX runtime features.

These specifications aim to provide a coherent structure for integrating AMX functionalities within the LLVM framework, ensuring optimal performance and compatibility across various hardware platforms.

## Conclusion
The outlined specifications are designed to address critical components of the AMX implementation. Adherence to these guidelines will ensure robust and efficient utilization of the AMX features within the LLVM project.