#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
dt00: dt00.$(OBJX) check.$(OBJX)
	@echo ------------ executing test $@
	@$(RM) a.$(EXESUFFIX)
	$(F90) $(LDFLAGS) dt00.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
	$(RUN2) a.$(EXESUFFIX)
dt00.$(OBJX): $(SRC)/dt00.f90
	$(F90) $(FFLAGS) $(SRC)/dt00.f90
build: dt00
run: ;
