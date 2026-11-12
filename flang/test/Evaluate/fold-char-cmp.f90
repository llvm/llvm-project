! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of character comparisons
module m1
  logical, parameter :: cases(*) = &
    [ "" == "", "" == "   " &
      , "aaa" == "aaa", "aaa" == "aaa ", "aaa" /= "aab" &
      , "aaa" <= "aaa", .not. "aaa" < "aaa", "aaa" < "aab", "aaa" >= "aaa" &
      , .not. "aaa" > "aaa", .not. "aaa" >= "aab" &
      , 4_"aaa" == 4_"aaa", 4_"aaa" == 4_"aaa ", 4_"aaa" /= 4_"aab" &
      , 4_"aaa" <= 4_"aaa", .not. 4_"aaa" < 4_"aaa", 4_"aaa" < 4_"aab", 4_"aaa" >= 4_"aaa" &
      , .not. 4_"aaa" > 4_"aaa", .not. 4_"aaa" >= 4_"aab" &
      , lle("aaa", "aaa"), .not. llt("aaa", "aaa"), llt("aaa", "aab"), lge("aaa", "aaa") &
      , .not. lgt("aaa", "aaa"), .not. lge("aaa", "aab") &
      , lle("", ""), .not. llt("", ""), lge("", ""), .not. lgt("", "") &
    ]
  logical, parameter :: test_cases = all(cases)
end module
