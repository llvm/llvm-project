#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fs22680.mk  ########


fs22680: run

build:  $(SRC)/fs22680.f90 
	-$(RM) fs22680.$(EXESUFFIX) core *.d *.mod
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/fs22680.f90 -o fs22680.$(EXESUFFIX)



run: 
	@echo ------------------------------------ executing test fs22680
	fs22680.$(EXESUFFIX)

verify: ;

fs22680.run: run

