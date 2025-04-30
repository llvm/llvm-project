#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in11  ########


in11: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in11.f90 $(SRC)/in11_expct.c fcheck.$(OBJX)
	-$(RM) in11.$(EXESUFFIX) in11.$(OBJX) in11_expct.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/in11_expct.c -o in11_expct.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in11.f90 -o in11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in11.$(OBJX) in11_expct.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in11
	in11.$(EXESUFFIX)

verify: ;

in11.run: run

