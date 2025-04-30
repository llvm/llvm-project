#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qaint  ########


qaint: run
	

build:  $(SRC)/qaint.f08
	-$(RM) qaint.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qaint.f08 -o qaint.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qaint.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qaint.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qaint 
	qaint.$(EXESUFFIX)

verify: ;


