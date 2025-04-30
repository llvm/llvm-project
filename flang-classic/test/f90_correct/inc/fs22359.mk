#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fs22359.mk  ########


fs22359: run

build:  $(SRC)/fs22359.f90
	-$(RM) fs22359.$(OBJX) core *.d *.mod
	@echo ------------------------------------ building test $@
	-$(FC) $(FFLAGS) $(LDFLAGS) $(SRC)/fs22359.f90 $(LIBS) -o fs22359.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test fs22359
	fs22359.$(EXESUFFIX)

verify: ;

fs22359.run: run

