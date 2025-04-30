#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp34  ########


qp34: run


build:  $(SRC)/qp34.f08
	-$(RM) qp34.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp34.f08 -o qp34.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp34.$(OBJX) check.$(OBJX) $(LIBS) -o qp34.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp34
	qp34.$(EXESUFFIX)

verify: ;
