#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI1vxm  ########


mmulI1vxm: run
	

build:  $(SRC)/mmulI1vxm.f90
	-$(RM) mmulI1vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI1vxm.f90 -o mmulI1vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI1vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI1vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI1vxm
	mmulI1vxm.$(EXESUFFIX)

verify: ;

mmulI1vxm.run: run

