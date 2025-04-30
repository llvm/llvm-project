#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR16vxm  ########


mmulR16vxm: run
	

build:  $(SRC)/mmulR16vxm.f08
	-$(RM) mmulR16vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR16vxm.f08 -o mmulR16vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR16vxm.$(OBJX) check_mod.$(OBJX) $(LIBS) -o mmulR16vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR16vxm
	mmulR16vxm.$(EXESUFFIX)

verify: ;

mmulR16vxm.run: run

