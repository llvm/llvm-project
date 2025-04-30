#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fs24236.mk  ########


fs24236: run

build:  $(SRC)/fs24236.f90 
	-$(RM) fs24236.$(EXESUFFIX) core *.d *.mod
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/fs24236.f90 -o fs24236.$(EXESUFFIX)



run: 
	@echo ------------------------------------ executing test fs24236
	fs24236.$(EXESUFFIX)

verify: ;

fs24236.run: run

