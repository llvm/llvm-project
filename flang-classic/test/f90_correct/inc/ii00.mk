#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ii00  ########


ii00: run
	

build:  $(SRC)/ii00.f90
	-$(RM) ii00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ii00.f90 -o ii00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ii00.$(OBJX) check.$(OBJX) $(LIBS) -o ii00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ii00
	ii00.$(EXESUFFIX)

verify: ;

ii00.run: run

