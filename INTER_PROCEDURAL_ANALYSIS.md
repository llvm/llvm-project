# Inter-Procedural Fixed-Point Analysis Implementation

## Overview
This implementation adds inter-procedural fixed-point analysis to One-Shot Bufferization (OSB) to handle recursive function call graphs.

## Key Changes

### 1. FuncAnalysisState Extension (`FuncBufferizableOpInterfaceImpl.h`)
- Added `callGraph`: Mapping from caller FuncOp to callees
- Added `reverseCallGraph`: Mapping from callee FuncOp to callers
- Added `iterationCount`: Tracks iteration count for each function
- Added `inWorklist`: Tracks functions in the worklist
- Added `hasRecursiveCall`: Flag indicating recursive calls detected

### 2. OneShotBufferizationOptions Extension (`OneShotAnalysis.h`)
- Added `maxFixedPointIterations`: Maximum iterations for fixed-point analysis (default: 10)

### 3. Core Functions (`OneShotModuleBufferize.cpp`)

#### `buildCallGraph(moduleOp, funcState)`
- Builds forward and reverse call graphs
- Detects recursive calls using DFS cycle detection

#### `initializeConservativeAssumptions(funcOps, funcState)`
- Initializes all tensor arguments as read and written
- Clears aliasing return values (assumes new allocations)

#### `runInterProceduralFixedPointAnalysis(moduleOp, funcOps, state, funcState, statistics)`
- Main fixed-point iteration loop
- Analyzes functions until convergence or max iterations
- Propagates changes to callers via worklist

#### `hasAnalysisChanged(funcOp, funcState, prevReadBbArgs, prevWrittenBbArgs, prevAliasingReturnVals)`
- Compares current analysis results with previous snapshot
- Returns true if any changes detected

### 4. Integration (`analyzeModuleOp`)
- Builds call graph first
- If recursive calls detected, uses fixed-point analysis
- Otherwise, uses existing DAG-based analysis

## Algorithm

1. **Build Call Graph**: Construct forward and reverse call graphs, detect cycles
2. **Initialize**: Set conservative assumptions (all args read/written)
3. **Fixed-Point Loop**:
   - Take snapshot of current state
   - Analyze all functions in worklist
   - Compare with previous snapshot
   - If changed, add callers to worklist for next iteration
   - Repeat until convergence or max iterations

## Usage

The fixed-point analysis is automatically enabled when:
- `bufferizeFunctionBoundaries` option is true
- Recursive calls are detected in the call graph

## Testing

Test cases should be added to verify:
- Simple recursive functions
- Mutually recursive functions (foo calls bar, bar calls foo)
- Convergence within iteration limit
- Correct alias analysis results

