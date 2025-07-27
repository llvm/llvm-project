# LLDB Fortran Support - Implementation Backlog

## Epic 1: Development Environment Setup
**Status**: 🔵 Ready  
**Goal**: Establish efficient development and testing environment

### Story 1.1: Minimal Build Configuration
- [ ] Create optimal CMake configuration for Fortran LLDB development
- [ ] Document build time and disk space requirements
- [ ] Verify build works with both Debug and Release configurations
- [ ] Test with different generators (Ninja, Make)

**Acceptance Criteria**:
- Build completes in under 30 minutes on standard hardware
- Only necessary targets are built (LLDB + minimal dependencies)
- Debug symbols available for development

**Test Cases**:
- Build with gfortran available
- Build with flang available
- Build without Fortran compilers (should still work)

### Story 1.2: Test Infrastructure Foundation
- [ ] Set up basic test structure in `lldb/test/API/lang/fortran/`
- [ ] Create simple Fortran test programs (.f90 files)
- [ ] Establish lit test configuration
- [ ] Add CMake integration for Fortran tests

**Acceptance Criteria**:
- Test directory structure follows LLDB conventions
- Tests can be run individually and in suites
- CI-friendly test execution

**Test Cases**:
- Run single Fortran test
- Run full Fortran test suite
- Tests work with both gfortran and flang

### Story 1.3: Continuous Integration Setup
- [ ] Create GitHub Actions workflow for Fortran tests
- [ ] Set up test matrix (multiple compilers, platforms)
- [ ] Configure test result reporting
- [ ] Add performance benchmarking

**Acceptance Criteria**:
- Tests run automatically on PR submission
- Results clearly visible in PR status
- Performance regressions are detected

---

## Epic 2: Phase 1 - Minimal Viable Plugin (MVP)
**Status**: 🔵 Ready  
**Goal**: Eliminate "no plugin" warning, basic language recognition

### Story 2.1: Plugin Registration Infrastructure (RED)
- [ ] Create `lldb/source/Plugins/Language/Fortran/` directory structure
- [ ] Implement basic `FortranLanguage.h` class declaration
- [ ] Add plugin registration in PluginManager
- [ ] Write failing test for language recognition

**Acceptance Criteria**:
- LLDB recognizes Fortran as a supported language
- No "unsupported language" warnings for Fortran files
- Plugin loads without errors

**Test Cases (RED phase)**:
```python
# Test: test_fortran_language_recognition.py
def test_fortran_language_detected(self):
    """Test that LLDB recognizes Fortran source files"""
    # This should FAIL initially
    self.expect("settings show target.language", 
                substrs=["fortran"])
```

### Story 2.2: Language Type Recognition (GREEN)
- [ ] Implement `GetLanguageType()` method
- [ ] Add support for all Fortran standards (F77, F90, F95, F2003, F2008, F2018)
- [ ] Implement source file extension recognition
- [ ] Make the RED test pass

**Acceptance Criteria**:
- LLDB correctly identifies Fortran language from DWARF
- All Fortran file extensions recognized (.f, .f90, .f95, etc.)
- Language enum values properly handled

**Test Cases (GREEN phase)**:
```python
def test_fortran_file_extensions(self):
    """Test recognition of various Fortran file extensions"""
    extensions = ['.f', '.f90', '.f95', '.f03', '.f08', '.f18']
    for ext in extensions:
        # Should now PASS
        self.assertTrue(is_fortran_file(f"test{ext}"))
```

### Story 2.3: Basic Plugin Methods (REFACTOR)
- [ ] Implement required Language base class methods
- [ ] Add proper error handling and logging
- [ ] Optimize plugin loading performance
- [ ] Add comprehensive documentation

**Acceptance Criteria**:
- All Language interface methods implemented
- Code follows LLVM coding standards
- No memory leaks or performance regressions

---

## Epic 3: Phase 2 - Type System Foundation
**Status**: ⏳ Waiting  
**Dependencies**: Epic 2 complete

### Story 3.1: DWARF Fortran Type Parsing (RED)
- [ ] Research DWARF format for Fortran types
- [ ] Write failing tests for COMPLEX type detection
- [ ] Write failing tests for CHARACTER type detection
- [ ] Write failing tests for Fortran array types

**Test Cases (RED phase)**:
```python
def test_complex_variable_display(self):
    """Test COMPLEX variable is displayed correctly"""
    # Should FAIL initially - shows as unknown type
    self.expect("frame variable complex_var",
                substrs=["(1.0, 2.0)"])  # Expected format
```

### Story 3.2: COMPLEX Type Support (GREEN)
- [ ] Implement COMPLEX type formatter
- [ ] Handle both single and double precision COMPLEX
- [ ] Support real and imaginary part access
- [ ] Make COMPLEX tests pass

**Acceptance Criteria**:
- COMPLEX variables display as "(real, imag)" format
- Individual real/imaginary parts accessible
- Both COMPLEX and DOUBLE COMPLEX supported

### Story 3.3: CHARACTER Type Support (GREEN)
- [ ] Implement CHARACTER type formatter
- [ ] Handle fixed-length CHARACTER variables
- [ ] Support CHARACTER array display
- [ ] Handle character length attribute from DWARF

**Acceptance Criteria**:
- CHARACTER variables show string content and length
- Null termination handled correctly
- CHARACTER arrays display properly

### Story 3.4: LOGICAL Type Support (GREEN)
- [ ] Implement LOGICAL type formatter
- [ ] Map LOGICAL values to .TRUE./.FALSE.
- [ ] Handle different LOGICAL kinds

**Acceptance Criteria**:
- LOGICAL variables display as .TRUE. or .FALSE.
- Different LOGICAL kinds supported

### Story 3.5: Basic Array Support (GREEN)
- [ ] Implement Fortran array bounds display
- [ ] Show array dimensions and size
- [ ] Handle 1-based indexing display
- [ ] Support multi-dimensional arrays

**Acceptance Criteria**:
- Arrays show bounds and dimensions
- 1-based indexing properly displayed
- Multi-dimensional arrays readable

### Story 3.6: Type System Refactoring (REFACTOR)
- [ ] Optimize type detection performance
- [ ] Consolidate common formatter code
- [ ] Add comprehensive type system tests
- [ ] Document type system architecture

---

## Epic 4: Phase 3 - Expression Evaluation
**Status**: ⏳ Waiting  
**Dependencies**: Epic 3 complete

### Story 4.1: Basic Expression Parser (RED)
- [ ] Write failing tests for simple Fortran expressions
- [ ] Research LLDB expression evaluation architecture
- [ ] Design Fortran expression parser interface

**Test Cases (RED phase)**:
```python
def test_fortran_expression_evaluation(self):
    """Test basic Fortran expression evaluation"""
    # Should FAIL initially
    self.expect("expression -- my_var + 1",
                substrs=["3"])  # If my_var = 2
```

### Story 4.2: Case-Insensitive Identifier Resolution (GREEN)
- [ ] Implement case-insensitive variable lookup
- [ ] Handle Fortran naming conventions
- [ ] Support compiler-specific name mangling

**Acceptance Criteria**:
- Variables accessible regardless of case in expression
- Proper symbol resolution for different compilers

### Story 4.3: Fortran Operators (GREEN)
- [ ] Implement Fortran-specific operators (**, .EQ., .NE., etc.)
- [ ] Handle operator precedence
- [ ] Support logical operators (.AND., .OR., .NOT.)

**Acceptance Criteria**:
- All Fortran operators work in expressions
- Correct operator precedence
- Logical operations produce LOGICAL results

### Story 4.4: Basic Intrinsic Functions (GREEN)
- [ ] Implement SIZE intrinsic function
- [ ] Implement LBOUND/UBOUND intrinsics
- [ ] Implement SHAPE intrinsic
- [ ] Add LEN intrinsic for CHARACTER

**Acceptance Criteria**:
- Array intrinsics return correct values
- Character intrinsics work properly
- Error handling for invalid arguments

---

## Epic 5: Phase 4 - Advanced Features
**Status**: ⏳ Waiting  
**Dependencies**: Epic 4 complete

### Story 5.1: Derived Type Support
- [ ] Parse Fortran derived types from DWARF
- [ ] Display derived type components
- [ ] Support type-bound procedures
- [ ] Handle inheritance

### Story 5.2: Module and Interface Support
- [ ] Support module variable access
- [ ] Handle USE association
- [ ] Support interface blocks
- [ ] Generic procedure resolution

### Story 5.3: Advanced Array Features
- [ ] Array sections and slicing
- [ ] Assumed-shape and deferred-shape arrays
- [ ] Allocatable and pointer arrays
- [ ] Array intrinsic functions

### Story 5.4: Parameterized Derived Types (PDTs)
- [ ] Support length and kind parameters
- [ ] Handle parameterized type instantiation
- [ ] Display parameter values

---

## Epic 6: Documentation and Community
**Status**: 🔵 Ready (parallel to development)

### Story 6.1: User Documentation
- [ ] Create Fortran debugging tutorial
- [ ] Document limitations and known issues
- [ ] Provide compiler-specific guidance
- [ ] Add troubleshooting guide

### Story 6.2: Developer Documentation
- [ ] Document plugin architecture decisions
- [ ] Create contribution guidelines
- [ ] Add design rationale documents
- [ ] Provide extending/modifying guide

### Story 6.3: Community Engagement
- [ ] Regular updates to issue #109119
- [ ] Engage with LLVM community for feedback
- [ ] Present at LLVM conferences
- [ ] Coordinate with Flang team

---

## Definition of Done

For each story to be considered complete:

### Code Quality
- [ ] All tests pass (unit + integration)
- [ ] Code review approved
- [ ] Performance impact assessed
- [ ] Memory leak testing passed

### Testing
- [ ] Red-Green-Refactor cycle completed
- [ ] Test coverage > 80% for new code
- [ ] Tests work with gfortran and flang
- [ ] Edge cases covered

### Documentation
- [ ] Code properly commented
- [ ] User-facing features documented
- [ ] Design decisions recorded
- [ ] Known limitations noted

### Integration
- [ ] CI tests pass
- [ ] No regressions in existing functionality
- [ ] Follows LLVM coding standards
- [ ] Ready for upstream submission

---

## Risk Register

### High-Risk Items
- **Performance Impact**: Adding language support could slow LLDB
  - *Mitigation*: Profile carefully, implement lazy loading
- **Compiler Compatibility**: gfortran vs flang differences
  - *Mitigation*: Test with both, abstract differences
- **DWARF Complexity**: Fortran DWARF can be complex
  - *Mitigation*: Start simple, add complexity incrementally

### Medium-Risk Items
- **Community Acceptance**: Changes may be rejected upstream
  - *Mitigation*: Engage early, follow guidelines strictly
- **Maintenance Burden**: Large codebase changes
  - *Mitigation*: Focus on clean, maintainable code

### Low-Risk Items
- **Build System Changes**: CMake modifications needed
  - *Mitigation*: Follow existing patterns

---

## Sprint Planning

### Sprint 1 (2 weeks): Environment Setup
- Epic 1 complete
- Story 2.1 started

### Sprint 2 (2 weeks): MVP Foundation
- Stories 2.1-2.2 complete
- Story 2.3 started

### Sprint 3 (2 weeks): MVP Complete
- Story 2.3 complete
- Story 3.1 started

### Sprint 4-6 (6 weeks): Type System
- Epic 3 complete

### Sprint 7-9 (6 weeks): Expression Evaluation
- Epic 4 complete

### Sprint 10+ (ongoing): Advanced Features
- Epic 5 progressive implementation

This backlog provides a clear roadmap following TDD principles while ensuring systematic progress toward full Fortran support in LLDB.