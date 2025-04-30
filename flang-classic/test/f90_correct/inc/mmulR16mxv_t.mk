#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16mxv_t  ########


mmulR16mxv_t: run
	

build:  $(SRC)/mmulR16mxv_t.f08
	-$(RM) mmulR16mxv_t.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16mxv_t.f08 -o mmulR16mxv_t.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16mxv_t.$(OBJX) check_mod.$(OBJX) $(LIBS) -o mmulR16mxv_t.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16mxv_t
	mmulR16mxv_t.$(EXESUFFIX)

verify: ;

mmulR16mxv_t.run: run

