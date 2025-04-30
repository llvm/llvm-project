#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qmax qmin  ########


qmax_min: run
	

build:  $(SRC)/qmax_min.f08
	-$(RM) qmax_min.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qmax_min.f08 -o qmax_min.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qmax_min.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qmax_min.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qmax_min 
	qmax_min.$(EXESUFFIX)

verify: ;


