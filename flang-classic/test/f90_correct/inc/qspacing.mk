#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qspacing  ########


qspacing: run
	

build:  $(SRC)/qspacing.f08
	-$(RM) qspacing.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qspacing.f08 -o qspacing.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qspacing.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qspacing.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qspacing 
	qspacing.$(EXESUFFIX)

verify: ;


