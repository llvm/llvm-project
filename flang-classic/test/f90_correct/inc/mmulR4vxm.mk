#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR4vxm  ########


mmulR4vxm: run
	

build:  $(SRC)/mmulR4vxm.f90
	-$(RM) mmulR4vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR4vxm.f90 -o mmulR4vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR4vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR4vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR4vxm
	mmulR4vxm.$(EXESUFFIX)

verify: ;

mmulR4vxm.run: run

