! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! CHECK: k=1_4
                        k=                                            1_99999999
     & 4
      end
