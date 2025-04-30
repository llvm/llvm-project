#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI8vxm  ########


mmulI8vxm: run
	

build:  $(SRC)/mmulI8vxm.f90
	-$(RM) mmulI8vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI8vxm.f90 -o mmulI8vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI8vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI8vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI8vxm
	mmulI8vxm.$(EXESUFFIX)

verify: ;

mmulI8vxm.run: run

