#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qhuge qnearest qepsilon  ########


qhuge_nearest_eps: run
	

build:  $(SRC)/qhuge_nearest_eps.f08
	-$(RM) qhuge_nearest_eps.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qhuge_nearest_eps.f08 -o qhuge_nearest_eps.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qhuge_nearest_eps.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qhuge_nearest_eps.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qhuge_nearest_eps 
	qhuge_nearest_eps.$(EXESUFFIX)

verify: ;


