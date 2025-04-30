#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC16vxm  ########


mmulC16vxm: run
	

build:  $(SRC)/mmulC16vxm.f90
	-$(RM) mmulC16vxm.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC16vxm.f90 -o mmulC16vxm.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC16vxm.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC16vxm.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC16vxm
	mmulC16vxm.$(EXESUFFIX)

verify: ;

mmulC16vxm.run: run

