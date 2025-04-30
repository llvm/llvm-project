#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qmod  ########


qmod: run
	

build:  $(SRC)/qmod.f08
	-$(RM) qmod.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qmod.f08 -o qmod.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qmod.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qmod.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qmod 
	qmod.$(EXESUFFIX)

verify: ;


