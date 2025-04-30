#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ba00  ########


ba00: run
	

build:  $(SRC)/ba00.f
	-$(RM) ba00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ba00.f -o ba00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ba00.$(OBJX) check.$(OBJX) $(LIBS) -o ba00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ba00
	ba00.$(EXESUFFIX)

verify: ;

ba00.run: run

