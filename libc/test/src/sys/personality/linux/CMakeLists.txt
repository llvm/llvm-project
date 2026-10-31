add_libc_testsuite(libc-sys-personality-tests)

add_libc_test(
  personality_test
  SUITE
    libc-sys-personality-tests
  SRCS
    personality_test.cpp
  DEPENDS
    libc.include.sys_personality
    libc.src.sys.personality.personality
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
