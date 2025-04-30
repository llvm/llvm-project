#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16_taxtb  ########


mmulR16_taxtb: run
	

build:  $(SRC)/mmulR16_taxtb.f08
	-$(RM) mmulR16_taxtb.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16_taxtb.f08  -o mmulR16_taxtb.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16_taxtb.$(OBJX) check_mod.$(OBJX) $(LIBS) -o mmulR16_taxtb.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16_taxtb
	mmulR16_taxtb.$(EXESUFFIX)

verify: ;

mmulR16_taxtb.run: run

