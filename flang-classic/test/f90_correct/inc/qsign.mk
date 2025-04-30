#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qsign  ########


qsign: run
	

build:  $(SRC)/qsign.f08
	-$(RM) qsign.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qsign.f08 -o qsign.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qsign.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qsign.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qsign 
	qsign.$(EXESUFFIX)

verify: ;


