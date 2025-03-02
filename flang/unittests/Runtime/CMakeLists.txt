add_flang_unittest(FlangRuntimeTests
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/AccessTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Allocatable.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/ArrayConstructor.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/BufferTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/CharacterTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/CommandTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Complex.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/CrashHandlerFixture.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Derived.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/ExternalIOTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Format.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Inquiry.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/ListInputTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/LogicalFormatTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Matmul.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/MatmulTranspose.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/MiscIntrinsic.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Namelist.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Numeric.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/NumericalFormatTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Pointer.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Ragged.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Random.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Reduction.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/RuntimeCrashTest.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Stop.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Support.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Time.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/TemporaryStack.cpp
  ${FLANG_RT_SOURCE_DIR}/unittests/Runtime/Transformational.cpp
)

target_link_libraries(FlangRuntimeTests
  PRIVATE
  flang_rt.runtime
)

target_compile_definitions(FlangRuntimeTests PRIVATE NOT_EXE="$<TARGET_FILE:not>")

add_subdirectory(CUDA)
