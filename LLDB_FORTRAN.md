# LLDB Fortran Language Support - Strategic Design Document

## Project Overview

This document outlines the strategic approach for implementing Fortran language support in LLDB (Issue #109119). The goal is to provide comprehensive debugging capabilities for Fortran programs, including variable inspection, expression evaluation, and Fortran-specific type support.

## Current State Analysis

### What Works Today
- Basic source-level debugging (step through lines, set breakpoints)
- DWARF debug information from Fortran compilers (gfortran, flang) contains rich metadata
- LLVM infrastructure already supports Fortran-specific DWARF tags and attributes

### Current Limitations
- Warning: "This version of LLDB has no plugin for the language 'fortran08'"
- Limited variable inspection capabilities
- No Fortran-specific expression evaluation
- No understanding of Fortran-specific types (COMPLEX, CHARACTER, etc.)
- Different symbol naming conventions across compilers (MAIN__ vs _QQmain)

## Technical Architecture

### Core Components Required

1. **Fortran Language Plugin** (`FortranLanguage` class)
   - Inherit from `Language` base class
   - Implement Fortran-specific language identification
   - Handle case-insensitive identifiers
   - Support multiple Fortran standards (77, 90, 95, 2003, 2008, 2018)

2. **Fortran Type System Integration**
   - COMPLEX type support (real and imaginary parts)
   - CHARACTER type with length specifications
   - Fortran array types with dimension information
   - LOGICAL type mapping
   - Derived type support
   - Parameterized derived types (PDTs)

3. **Expression Parser Enhancement**
   - Fortran syntax support in expression evaluator
   - Intrinsic function support (SIZE, SHAPE, LBOUND, UBOUND, etc.)
   - Fortran operators (**, .EQ., .NE., etc.)
   - Case-insensitive variable name resolution

4. **Symbol Resolution**
   - Handle compiler-specific name mangling (gfortran vs flang)
   - Main program entry point identification
   - Module and subprogram name resolution
   - Common block variable handling

### Plugin Architecture Integration

Following the established pattern in LLDB:
```
lldb/source/Plugins/Language/Fortran/
├── CMakeLists.txt
├── FortranLanguage.h
├── FortranLanguage.cpp
└── FortranFormatters.cpp
```

**Plugin Registration**: Use `LLDB_PLUGIN_DEFINE(FortranLanguage)` macro and register with PluginManager following the pattern from CPlusPlusLanguage and ObjCLanguage.

### DWARF Integration Points

Leverage existing Fortran-specific DWARF features:
- `DW_TAG_string_type` for CHARACTER variables
- `DW_AT_elemental`, `DW_AT_pure`, `DW_AT_recursive` for procedure attributes
- Fortran array type information
- Complex number representation

## Implementation Strategy

### Phase 1: Minimal Viable Plugin (MVP)
**Goal**: Eliminate the "no plugin" warning and basic functionality

**Deliverables**:
- Basic `FortranLanguage` plugin registration
- Language type recognition for all Fortran standards
- Source file extension recognition (.f, .f90, .f95, .f03, .f08, .f18)
- Minimal test infrastructure

**Testing Strategy**:
- Unit tests for language recognition
- Basic regression tests with simple Fortran programs
- CI integration with both gfortran and flang

### Phase 2: Type System Foundation
**Goal**: Proper display of Fortran variables

**Deliverables**:
- COMPLEX type formatter
- CHARACTER type formatter with length display
- LOGICAL type formatter
- Basic array display
- Enhanced DWARF parsing for Fortran types

**Testing Strategy**:
- Type-specific test programs
- Variable inspection tests
- Memory layout verification

### Phase 3: Expression Evaluation
**Goal**: Enable Fortran expression evaluation in debugger

**Deliverables**:
- Fortran expression parser
- Basic intrinsic function support
- Case-insensitive identifier resolution
- Fortran operator support

**Testing Strategy**:
- Expression evaluation test suite
- Intrinsic function tests
- Operator precedence tests

### Phase 4: Advanced Features
**Goal**: Complete Fortran debugging experience

**Deliverables**:
- Derived type support
- Module debugging
- Advanced array operations
- Fortran-specific exception handling
- Comprehensive documentation

## Build System Integration

### Selective Building Strategy
Following LLVM best practices for focused development:

```bash
# Recommended LLDB build for Fortran development
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;lldb;flang" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLDB_INCLUDE_TESTS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_USE_LINKER=lld
```

**Note**: Using `RelWithDebInfo` provides better performance than `Debug` while maintaining debug symbols. Use `Debug` only when debugging LLDB itself.

### Test-Driven Development Workflow
1. Write failing test for new functionality
2. Implement minimal code to pass test
3. Refactor and optimize
4. Ensure all existing tests still pass

### Testing Infrastructure
- **Unit Tests**: `lldb/unittests/Language/Fortran/`
- **Integration Tests**: `lldb/test/API/lang/fortran/`
- **Lit Tests**: `lldb/test/Shell/lang/fortran/`

## Risk Assessment & Mitigation

### Technical Risks
1. **Compiler Divergence**: Different Fortran compilers (gfortran vs flang) may generate incompatible DWARF
   - *Mitigation*: Test with both compilers, abstract common functionality
   
2. **Complex Type System**: Fortran has unique features not present in C/C++
   - *Mitigation*: Phase implementation, start with basic types
   
3. **Performance Impact**: Additional language support may slow down LLDB
   - *Mitigation*: Profile performance, use lazy loading where possible

### Project Risks
1. **Large Codebase**: LLVM/LLDB is enormous and complex
   - *Mitigation*: Focus on minimal builds, selective testing
   
2. **Community Integration**: Changes need upstream acceptance
   - *Mitigation*: Engage with LLVM community early, follow contribution guidelines

## Success Metrics

### Short-term (Phase 1-2)
- No "unsupported language" warnings for Fortran files
- Basic variable inspection works for primitive types
- CI tests pass consistently

### Medium-term (Phase 3)
- Expression evaluation works for simple Fortran expressions
- COMPLEX and CHARACTER types display correctly
- Arrays can be inspected element by element

### Long-term (Phase 4)
- Full Fortran debugging feature parity with C/C++
- Support for derived types and modules
- Community adoption and upstream integration

## Dependencies

### External Dependencies
- LLVM/Clang infrastructure (already present)
- DWARF standard support for Fortran
- Test Fortran compilers (gfortran, flang)

### Internal Dependencies
- LLDB plugin architecture
- Expression evaluation framework
- Type system infrastructure
- Testing framework

## Next Steps

1. Set up minimal build configuration for rapid iteration
2. Create basic test infrastructure for Fortran
3. Implement Phase 1 MVP (basic plugin registration)
4. Establish CI pipeline with Fortran test cases
5. Begin Phase 2 implementation (type system)

This strategic approach ensures incremental progress while maintaining code quality and test coverage throughout the development process.