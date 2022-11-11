! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror

!PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg1 has length 64, which is greater than the maximum name length 63
program aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg1

  !PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg2 has length 64, which is greater than the maximum name length 63
  integer :: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg2

  integer :: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg

  !PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg3 has length 64, which is greater than the maximum name length 63
  call aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg3

end
