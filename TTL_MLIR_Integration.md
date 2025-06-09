# TTL MLIR Integration

## Project Overview
This project aims to integrate TTL (Template Tiling Library) with MLIR to create an optimized pipeline from C code to TTL-optimized C code. The pipeline includes affine loop tiling and dialect conversions, with a focus on optimizing operations like sigmoid.

## Current Pipeline
```
C code with TTL DSL → MLIR → Optimized MLIR → EmitC → C code
```

## Technical Implementation

### Version Compatibility
- Using LLVM 20 for MLIR pipeline
- Polygeist (C → MLIR) is on LLVM 18
- Solution: Manually removing incompatible parts
- This is a manageable limitation for now

### Type System Integration
- Minor issue with unrealized conversion casts
- Can be fixed with a simple pass if needed
- Not a critical blocker

### TTL Integration Strategy
Two possible approaches:
1. Generate direct function calls to TTL's existing functions
2. Create a TTL dialect (if needed)
- Currently leaning towards function calls for simplicity
- Decision pending based on future requirements

## Next Steps

### 1. Frontend Definition
- Define Polygeist as the frontend
- Its output will feed into TTL optimizer passes (like tiling)
- Currently supporting minimal 2D loops and array access
- Will expand TTL DSL features in the frontend

### 2. Backend Generation
- Develop pipeline to generate TTL-specific code
- Focus on efficient memory operations and tiling

### 3. TTL DSL Development
- Currently minimal: 2D loops and array access
- Will expand based on requirements
- Starting with sigmoid as a test case

### 4. Immediate Focus
- Optimizing sigmoid function
- Using it as a test case for the complete pipeline
- Will use learnings to expand to other operations

## Technical Decisions
- Keeping things simple with function calls rather than new dialect
- Managing version compatibility manually for now
- Type conversion issues are minor and can be addressed if needed

## Current Limitations
1. Version mismatch between Polygeist and MLIR pipeline
2. Minimal TTL DSL features in frontend
3. Focus on sigmoid optimization only

## Future Work
1. Expand TTL DSL features
2. Add more optimization passes
3. Support more complex operations
4. Evaluate need for TTL dialect
5. Consider automating version compatibility fixes 