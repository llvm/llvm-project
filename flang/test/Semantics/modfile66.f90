! RUN: %flang_fc1 -fsyntax-only %S/Inputs/modfile66.cuf && %flang_fc1 -fsyntax-only %s
use usereal2 ! valid since x is not used
end
