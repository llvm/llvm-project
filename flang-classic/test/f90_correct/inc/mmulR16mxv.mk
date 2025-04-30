#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16mxv  ########


mmulR16mxv: run
	

build:  $(SRC)/mmulR16mxv.f08
	-$(RM) mmulR16mxv.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16mxv.f08 -o mmulR16mxv.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16mxv.$(OBJX) check_mod.$(OBJX) $(LIBS) -o mmulR16mxv.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16mxv
	mmulR16mxv.$(EXESUFFIX)

verify: ;

mmulR16mxv.run: run

