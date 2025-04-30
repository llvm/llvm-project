#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test async_1 ########


async_1: run

build:  $(SRC)/async_1.f90
	-$(RM) async_1.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/async_1.f90 -o async_1.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) async_1.$(OBJX) check.$(OBJX) $(LIBS) -o async_1.$(EXESUFFIX)


run:
	$(RM) ./*.txt
	@echo ------------------------------------ executing test async_1
	$(CP) $(SRC)/async_files/*.txt .
	@chmod 777 ./*.txt
	async_1.$(EXESUFFIX)

verify: ;
