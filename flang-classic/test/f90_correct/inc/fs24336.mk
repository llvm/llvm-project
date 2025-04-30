#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fs24336.mk  ########


fs24336: run

build:  $(SRC)/fs24336_module.f90 $(SRC)/fs24336.f90
	-$(RM) fs24336_module.$(OBJX) fs24336.$(OBJX) core *.d *.mod
	@echo ------------------------------------ building test $@
	-$(FC) -v $(FFLAGS) $(LDFLAGS) $(SRC)/fs24336_module.f90 $(SRC)/fs24336.f90 $(LIBS) -o fs24336.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test fs24336
	fs24336.$(EXESUFFIX)

verify: ;

fs24336.run: run

