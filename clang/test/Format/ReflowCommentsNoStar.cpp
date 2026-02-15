// RUN: clang-format -style="{ColumnLimit: 80, ReflowComments: true, ReflowCommentsNoStar: true}" %S/ReflowCommentsNoStarInput.cpp > %t
// RUN: diff %t %S/ReflowCommentsNoStarExpected.cpp