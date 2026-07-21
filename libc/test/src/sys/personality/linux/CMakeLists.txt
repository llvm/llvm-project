add_custom_target(libc_sys_personality_unittests)

add_libc_unittest(
  personality_test
  SUITE
    libc_sys_personality_unittests
  SRCS
    personality_test.cpp
  DEPENDS
    libc.include.sys_personality
    libc.src.sys.personality.personality
    libc.src.errno.errno
    libc.test.UnitTest.ErrnoCheckingTest
    libc.test.UnitTest.ErrnoSetterMatcher
)
