#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test stoq  ########


qp70: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp70.f08 fcheck.$(OBJX)
	-$(RM) qp70.$(EXESUFFIX) qp70.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp70.f08 -o qp70.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp70.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp70.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp70
	qp70.$(EXESUFFIX)

verify: ;

qp70.run: run

