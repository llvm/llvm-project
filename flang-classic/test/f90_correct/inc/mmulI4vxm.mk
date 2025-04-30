#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulI4vxm  ########


mmulI4vxm: run
	

build:  $(SRC)/mmulI4vxm.f90
	-$(RM) mmulI4vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulI4vxm.f90 -o mmulI4vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulI4vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulI4vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulI4vxm
	mmulI4vxm.$(EXESUFFIX)

verify: ;

mmulI4vxm.run: run

