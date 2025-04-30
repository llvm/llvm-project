#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16mxm_t  ########


mmulR16mxm_t: run
	

build:  $(SRC)/mmulR16mxm_t.f08
	-$(RM) mmulR16mxm_t.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16mxm_t.f08 -o mmulR16mxm_t.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16mxm_t.$(OBJX) check_mod.$(OBJX) $(LIBS) -o mmulR16mxm_t.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16mxm_t
	mmulR16mxm_t.$(EXESUFFIX)

verify: ;

mmulR16mxm_t.run: run

