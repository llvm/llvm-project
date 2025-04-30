#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR8vxm  ########


mmulR8vxm: run
	

build:  $(SRC)/mmulR8vxm.f90
	-$(RM) mmulR8vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR8vxm.f90 -o mmulR8vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR8vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR8vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR8vxm
	mmulR8vxm.$(EXESUFFIX)

verify: ;

mmulR8vxm.run: run

