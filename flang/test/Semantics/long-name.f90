! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -pedantic

!PORTABILITY: AAAAAAAAAABBBBBBBBBBCCCCCCCCCCDDDDDDDDDDEEEEEEEEEEFFFFFFFFFFGGG1 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
program aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg1

  !PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg2 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
  integer :: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg2

  integer :: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg

  !PORTABILITY: aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg3 has length 64, which is greater than the maximum name length 63 [-Wlong-names]
  call aaaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffggg3

end
