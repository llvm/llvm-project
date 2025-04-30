#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC8vxm  ########


mmulC8vxm: run
	

build:  $(SRC)/mmulC8vxm.f90
	-$(RM) mmulC8vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC8vxm.f90 -o mmulC8vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC8vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC8vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC8vxm
	mmulC8vxm.$(EXESUFFIX)

verify: ;

mmulC8vxm.run: run

