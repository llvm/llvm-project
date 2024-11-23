/*
 * Copyright (c) 2020 Bitdefender
 * SPDX-License-Identifier: Apache-2.0
 */
#include "include/bddisasm_crt.h"
#include "../inc/bddisasm.h"

// The table definitions.
#include "include/bdx86_tabledefs.h"

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P) ((void)(P))
#endif


static const ND_UINT8 gDispsizemap16[4][8] =
{
    { 0, 0, 0, 0, 0, 0, 2, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1 },
    { 2, 2, 2, 2, 2, 2, 2, 2 },
    { 0, 0, 0, 0, 0, 0, 0, 0 },
};

static const ND_UINT8 gDispsizemap[4][8] =
{
    { 0, 0, 0, 0, 0, 4, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1 },
    { 4, 4, 4, 4, 4, 4, 4, 4 },
    { 0, 0, 0, 0, 0, 0, 0, 0 },
};


//
// NdGetVersion
//
void
NdGetVersion(
    ND_UINT32 *Major,
    ND_UINT32 *Minor,
    ND_UINT32 *Revision,
    const char **BuildDate,
    const char **BuildTime
    )
{
    if (ND_NULL != Major)
    {
        *Major = DISASM_VERSION_MAJOR;
    }

    if (ND_NULL != Minor)
    {
        *Minor = DISASM_VERSION_MINOR;
    }

    if (ND_NULL != Revision)
    {
        *Revision = DISASM_VERSION_REVISION;
    }

//
// Do not use __TIME__ and __DATE__ macros when compiling against a kernel tree.
//
#if defined(__KERNEL__)

    if (ND_NULL != BuildDate)
    {
        *BuildDate = (char *)ND_NULL;
    }

    if (ND_NULL != BuildTime)
    {
        *BuildTime = (char *)ND_NULL;
    }

#else

    if (ND_NULL != BuildDate)
    {
        *BuildDate = __DATE__;
    }

    if (ND_NULL != BuildTime)
    {
        *BuildTime = __TIME__;
    }

#endif

}

//
// NdFetchData
//
static ND_UINT64
NdFetchData(
    const ND_UINT8 *Buffer,
    ND_UINT8 Size
    )
{
    switch (Size) 
    {
    case 1:
        return ND_FETCH_8(Buffer);
    case 2:
        return ND_FETCH_16(Buffer);
    case 4:
        return ND_FETCH_32(Buffer);
    case 8:
        return ND_FETCH_64(Buffer);
    default:
        return 0;
    }
}


//
// NdFetchXop
//
static NDSTATUS
NdFetchXop(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // Offset points to the 0x8F XOP prefix.
    // One more byte has to follow, the modrm or the second XOP byte.
    RET_GT((ND_SIZET)Offset + 2, Size, ND_STATUS_BUFFER_TOO_SMALL);

    if (((Code[Offset + 1] & 0x1F) >= 8))
    {
        // XOP found, make sure the third byte is here.
        RET_GT((ND_SIZET)Offset + 3, Size, ND_STATUS_BUFFER_TOO_SMALL);

        // Make sure we don't have any other prefix.
        if (Instrux->HasOpSize || 
            Instrux->HasRepnzXacquireBnd || 
            Instrux->HasRepRepzXrelease || 
            Instrux->HasRex || 
            Instrux->HasRex2)
        {
            return ND_STATUS_XOP_WITH_PREFIX;
        }

        // Fill in XOP info.
        Instrux->HasXop = ND_TRUE;
        Instrux->EncMode = ND_ENCM_XOP;
        Instrux->Xop.Xop[0] = Code[Offset];
        Instrux->Xop.Xop[1] = Code[Offset + 1];
        Instrux->Xop.Xop[2] = Code[Offset + 2];

        Instrux->Exs.w = Instrux->Xop.w;
        Instrux->Exs.r = (ND_UINT32)~Instrux->Xop.r;
        Instrux->Exs.x = (ND_UINT32)~Instrux->Xop.x;
        Instrux->Exs.b = (ND_UINT32)~Instrux->Xop.b;
        Instrux->Exs.l = Instrux->Xop.l;
        Instrux->Exs.v = (ND_UINT32)~Instrux->Xop.v;
        Instrux->Exs.m = Instrux->Xop.m;
        Instrux->Exs.p = Instrux->Xop.p;

        // if we are in non 64 bit mode, we must make sure that none of the extended registers are being addressed.
        if (Instrux->DefCode != ND_CODE_64)
        {
            // Xop.R and Xop.X must be 1 (inverted).
            if ((Instrux->Exs.r | Instrux->Exs.x) == 1)
            {
                return ND_STATUS_INVALID_ENCODING_IN_MODE;
            }

            // Xop.V must be less than 8.
            if ((Instrux->Exs.v & 0x8) == 0x8)
            {
                return ND_STATUS_INVALID_ENCODING_IN_MODE;
            }

            // Xop.B is ignored, so we force it to 0.
            Instrux->Exs.b = 0;
        }

        // Update Instrux length & offset, and make sure we don't exceed 15 bytes.
        Instrux->Length += 3;
        if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
        {
            return ND_STATUS_INSTRUCTION_TOO_LONG;
        }
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchVex2
//
static NDSTATUS
NdFetchVex2(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // One more byte has to follow, the modrm or the second VEX byte.
    RET_GT((ND_SIZET)Offset + 2, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // VEX is available only in 32 & 64 bit mode.
    if ((ND_CODE_64 == Instrux->DefCode) || ((Code[Offset + 1] & 0xC0) == 0xC0))
    {
        // Make sure we don't have any other prefix.
        if (Instrux->HasOpSize || 
            Instrux->HasRepnzXacquireBnd ||
            Instrux->HasRepRepzXrelease || 
            Instrux->HasRex || 
            Instrux->HasRex2 || 
            Instrux->HasLock)
        {
            return ND_STATUS_VEX_WITH_PREFIX;
        }

        // Fill in VEX2 info.
        Instrux->VexMode = ND_VEXM_2B;
        Instrux->HasVex = ND_TRUE;
        Instrux->EncMode = ND_ENCM_VEX;
        Instrux->Vex2.Vex[0] = Code[Offset];
        Instrux->Vex2.Vex[1] = Code[Offset + 1];

        Instrux->Exs.m = 1; // For VEX2 instructions, always use the second table.
        Instrux->Exs.r = (ND_UINT32)~Instrux->Vex2.r;
        Instrux->Exs.v = (ND_UINT32)~Instrux->Vex2.v;
        Instrux->Exs.l = Instrux->Vex2.l;
        Instrux->Exs.p = Instrux->Vex2.p;

        // Update Instrux length & offset, and make sure we don't exceed 15 bytes.
        Instrux->Length += 2;
        if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
        {
            return ND_STATUS_INSTRUCTION_TOO_LONG;
        }
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchVex3
//
static NDSTATUS
NdFetchVex3(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // One more byte has to follow, the modrm or the second VEX byte.
    RET_GT((ND_SIZET)Offset + 2, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // VEX is available only in 32 & 64 bit mode.
    if ((ND_CODE_64 == Instrux->DefCode) || ((Code[Offset + 1] & 0xC0) == 0xC0))
    {
        // VEX found, make sure the third byte is here.
        RET_GT((ND_SIZET)Offset + 3, Size, ND_STATUS_BUFFER_TOO_SMALL);

        // Make sure we don't have any other prefix.
        if (Instrux->HasOpSize || 
            Instrux->HasRepnzXacquireBnd ||
            Instrux->HasRepRepzXrelease || 
            Instrux->HasRex || 
            Instrux->HasRex2 || 
            Instrux->HasLock)
        {
            return ND_STATUS_VEX_WITH_PREFIX;
        }

        // Fill in XOP info.
        Instrux->VexMode = ND_VEXM_3B;
        Instrux->HasVex = ND_TRUE;
        Instrux->EncMode = ND_ENCM_VEX;
        Instrux->Vex3.Vex[0] = Code[Offset];
        Instrux->Vex3.Vex[1] = Code[Offset + 1];
        Instrux->Vex3.Vex[2] = Code[Offset + 2];

        Instrux->Exs.r = (ND_UINT32)~Instrux->Vex3.r;
        Instrux->Exs.x = (ND_UINT32)~Instrux->Vex3.x;
        Instrux->Exs.b = (ND_UINT32)~Instrux->Vex3.b;
        Instrux->Exs.m = Instrux->Vex3.m;
        Instrux->Exs.w = Instrux->Vex3.w;
        Instrux->Exs.v = (ND_UINT32)~Instrux->Vex3.v;
        Instrux->Exs.l = Instrux->Vex3.l;
        Instrux->Exs.p = Instrux->Vex3.p;

        // Do validations in case of VEX outside 64 bits.
        if (Instrux->DefCode != ND_CODE_64)
        {
            // Vex.R and Vex.X have been tested by the initial if.

            // Vex.vvvv must be less than 8.
            Instrux->Exs.v &= 7;

            // Vex.B is ignored, so we force it to 0.
            Instrux->Exs.b = 0;
        }

        // Update Instrux length & offset, and make sure we don't exceed 15 bytes.
        Instrux->Length += 3;
        if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
        {
            return ND_STATUS_INSTRUCTION_TOO_LONG;
        }
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchEvex
//
static NDSTATUS
NdFetchEvex(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // One more byte has to follow, the modrm or the second VEX byte.
    RET_GT((ND_SIZET)Offset + 2, Size, ND_STATUS_BUFFER_TOO_SMALL);

    if ((ND_CODE_64 != Instrux->DefCode) && ((Code[Offset + 1] & 0xC0) != 0xC0))
    {
        // BOUND instruction in non-64 bit mode, not EVEX.
        return ND_STATUS_SUCCESS;
    }

    // EVEX found, make sure all the bytes are present. At least 4 bytes in total must be present.
    RET_GT((ND_SIZET)Offset + 4, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // This is EVEX.
    Instrux->HasEvex = ND_TRUE;
    Instrux->EncMode = ND_ENCM_EVEX;
    Instrux->Evex.Evex[0] = Code[Offset + 0];
    Instrux->Evex.Evex[1] = Code[Offset + 1];
    Instrux->Evex.Evex[2] = Code[Offset + 2];
    Instrux->Evex.Evex[3] = Code[Offset + 3];

    // Legacy prefixes are not accepted with EVEX.
    if (Instrux->HasOpSize || 
        Instrux->HasRepnzXacquireBnd || 
        Instrux->HasRepRepzXrelease || 
        Instrux->HasRex || 
        Instrux->HasRex2 ||
        Instrux->HasLock)
    {
        return ND_STATUS_EVEX_WITH_PREFIX;
    }

    // Do the opcode independent checks. Opcode dependent checks are done when decoding each instruction.
    if (Instrux->Evex.m == 0)
    {
        return ND_STATUS_INVALID_ENCODING;
    }

    // Check map. Maps 4 & 7 are allowed only if APX is enabled.
    if (Instrux->Evex.m == 4 || Instrux->Evex.m == 7)
    {
        if (!(Instrux->FeatMode & ND_FEAT_APX))
        {
            return ND_STATUS_INVALID_ENCODING;
        }
    }


    // Fill in the generic extension bits. We initially optimistically fill in all possible values.
    // Once we determine the opcode and, subsequently, the EVEX extension mode, we will do further 
    // validations, and reset unused fields to 0.
    Instrux->Exs.r = (ND_UINT32)~Instrux->Evex.r;
    Instrux->Exs.x = (ND_UINT32)~Instrux->Evex.x;
    Instrux->Exs.b = (ND_UINT32)~Instrux->Evex.b;
    Instrux->Exs.rp = (ND_UINT32)~Instrux->Evex.rp;
    Instrux->Exs.x4 = (ND_UINT32)~Instrux->Evex.u;
    Instrux->Exs.b4 = Instrux->Evex.b4;
    Instrux->Exs.m = Instrux->Evex.m;
    Instrux->Exs.w = Instrux->Evex.w;
    Instrux->Exs.v = (ND_UINT32)~Instrux->Evex.v;
    Instrux->Exs.vp = (ND_UINT32)~Instrux->Evex.vp;
    Instrux->Exs.p = Instrux->Evex.p;

    Instrux->Exs.z = Instrux->Evex.z;
    Instrux->Exs.l = Instrux->Evex.l;
    Instrux->Exs.bm = Instrux->Evex.bm;
    Instrux->Exs.k = Instrux->Evex.a;

    // EVEX extensions. The fields are undefined if the encoding does not use them.
    Instrux->Exs.nf = (Instrux->Evex.Evex[3] >> 2) & 1;
    Instrux->Exs.nd = (Instrux->Evex.Evex[3] >> 4) & 1;
    Instrux->Exs.sc = (Instrux->Evex.Evex[3] & 0xF);

    // Do EVEX validations outside 64 bits mode.
    if (ND_CODE_64 != Instrux->DefCode)
    {
        // Evex.R and Evex.X must be 1. If they're not, we have BOUND instruction. This is checked in the
        // first if. Note that they are inverted inside the Evex prefix.
        Instrux->Exs.r = 0;
        Instrux->Exs.x = 0;

        // Evex.B is ignored, so we force it to 0.
        Instrux->Exs.b = 0;

        // Evex.R' is ignored, so we force it to 0.
        Instrux->Exs.rp = 0;

        // Evex.B4 & Evex.X4 are ignored, so we force them to 0.
        Instrux->Exs.b4 = Instrux->Exs.x4 = 0;

        // High bit inside Evex.VVVV is ignored, so we force it to 0.
        Instrux->Exs.v &= 0x7;

        // Evex.V' must be 1 (negated to 0) in 32-bit mode.
        if (Instrux->Exs.vp == 1)
        {
            return ND_STATUS_BAD_EVEX_V_PRIME;
        }
    }

    // Update Instrux length & offset, and make sure we don't exceed 15 bytes.
    Instrux->Length += 4;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchRex2
//
static NDSTATUS
NdFetchRex2(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    if (ND_CODE_64 != Instrux->DefCode)
    {
        // AAD instruction outside 64-bit mode.
        return ND_STATUS_SUCCESS;
    }

    if (!(Instrux->FeatMode & ND_FEAT_APX))
    {
        // APX not enabled, #UD.
        return ND_STATUS_SUCCESS;
    }

    // One more byte has to follow.
    RET_GT((ND_SIZET)Offset + 2, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // This is REX2.
    Instrux->HasRex2 = ND_TRUE;
    Instrux->EncMode = ND_ENCM_LEGACY;
    Instrux->Rex2.Rex2[0] = Code[Offset + 0];
    Instrux->Rex2.Rex2[1] = Code[Offset + 1];

    // REX illegal with REX2.
    if (Instrux->HasRex)
    {
        return ND_STATUS_INVALID_PREFIX_SEQUENCE;
    }

    // Fill in the generic extension bits
    Instrux->Exs.r = Instrux->Rex2.r3;
    Instrux->Exs.rp = Instrux->Rex2.r4;
    Instrux->Exs.x = Instrux->Rex2.x3;
    Instrux->Exs.x4 = Instrux->Rex2.x4;
    Instrux->Exs.b = Instrux->Rex2.b3;
    Instrux->Exs.b4 = Instrux->Rex2.b4;
    Instrux->Exs.w = Instrux->Rex2.w;

    // Update Instrux length & offset, and make sure we don't exceed 15 bytes.
    Instrux->Length += 2;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchPrefixes
//
static NDSTATUS
NdFetchPrefixes(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    NDSTATUS status;
    ND_BOOL morePrefixes;
    ND_UINT8 prefix;

    morePrefixes = ND_TRUE;

    while (morePrefixes)
    {
        morePrefixes = ND_FALSE;

        RET_GT((ND_SIZET)Offset + 1, Size, ND_STATUS_BUFFER_TOO_SMALL);

        prefix = Code[Offset];

        // Speedup: if the current byte is not a prefix of any kind, leave now. This will be the case most of the times.
        if (ND_PREF_CODE_NONE == gPrefixesMap[prefix])
        {
            status = ND_STATUS_SUCCESS;
            goto done_prefixes;
        }

        if (ND_PREF_CODE_STANDARD == gPrefixesMap[prefix])
        {
            switch (prefix)
            {
            case ND_PREFIX_G0_LOCK:
                Instrux->HasLock = ND_TRUE;
                morePrefixes = ND_TRUE;
                break;
            case ND_PREFIX_G1_REPE_REPZ:
                Instrux->Rep = ND_PREFIX_G1_REPE_REPZ;
                Instrux->HasRepRepzXrelease = ND_TRUE;
                morePrefixes = ND_TRUE;
                break;
            case ND_PREFIX_G1_REPNE_REPNZ:
                Instrux->Rep = ND_PREFIX_G1_REPNE_REPNZ;
                Instrux->HasRepnzXacquireBnd = ND_TRUE;
                morePrefixes = ND_TRUE;
                break;
            case ND_PREFIX_G2_SEG_CS:
            case ND_PREFIX_G2_SEG_SS:
            case ND_PREFIX_G2_SEG_DS:
            case ND_PREFIX_G2_SEG_ES:
            case ND_PREFIX_G2_SEG_FS:
            case ND_PREFIX_G2_SEG_GS:
                if (ND_CODE_64 == Instrux->DefCode)
                {
                    if (prefix == ND_PREFIX_G2_SEG_FS || 
                        prefix == ND_PREFIX_G2_SEG_GS)
                    {
                        // The last FS/GS is always used, if present.
                        Instrux->Seg = prefix;
                        Instrux->HasSeg = ND_TRUE;
                    }
                    else if (prefix == ND_PREFIX_G2_NO_TRACK && 
                        Instrux->Seg != ND_PREFIX_G2_SEG_FS &&
                        Instrux->Seg != ND_PREFIX_G2_SEG_GS)
                    {
                        // The Do Not Track prefix is considered only if there isn't a FS/GS prefix.
                        Instrux->Seg = prefix;
                        Instrux->HasSeg = ND_TRUE;
                    }
                    else if (Instrux->Seg != ND_PREFIX_G2_SEG_FS && 
                        Instrux->Seg != ND_PREFIX_G2_SEG_GS &&
                        Instrux->Seg != ND_PREFIX_G2_NO_TRACK)
                    {
                        // All other prefixes are considered if Do Not Track, FS, GS are not present.
                        Instrux->Seg = prefix;
                        Instrux->HasSeg = ND_TRUE;
                    }
                }
                else
                {
                    Instrux->Seg = prefix;
                    Instrux->HasSeg = ND_TRUE;
                }
                morePrefixes = ND_TRUE;
                break;
            case ND_PREFIX_G3_OPERAND_SIZE:
                Instrux->HasOpSize = ND_TRUE;
                morePrefixes = ND_TRUE;
                break;
            case ND_PREFIX_G4_ADDR_SIZE:
                Instrux->HasAddrSize = ND_TRUE;
                morePrefixes = ND_TRUE;
                break;
            default:
                break;
            }
        }

        // REX must precede the opcode byte. However, if one or more other prefixes are present, the instruction
        // will still decode & execute properly, but REX will be ignored.
        if (morePrefixes && Instrux->HasRex)
        {
            Instrux->HasRex = ND_FALSE;
            Instrux->Rex.Rex = 0;
            Instrux->Exs.w = 0;
            Instrux->Exs.r = 0;
            Instrux->Exs.x = 0;
            Instrux->Exs.b = 0;
        }

        // Check for REX.
        if ((ND_CODE_64 == Instrux->DefCode) && (ND_PREF_CODE_REX == gPrefixesMap[prefix]))
        {
            Instrux->HasRex = ND_TRUE;
            Instrux->Rex.Rex = prefix;
            Instrux->Exs.w = Instrux->Rex.w;
            Instrux->Exs.r = Instrux->Rex.r;
            Instrux->Exs.x = Instrux->Rex.x;
            Instrux->Exs.b = Instrux->Rex.b;
            morePrefixes = ND_TRUE;
        }

        // We have found prefixes, update the instruction length and the current offset.
        if (morePrefixes)
        {
            Instrux->Length++, Offset++;
            if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
            {
                return ND_STATUS_INSTRUCTION_TOO_LONG;
            }
        }
    }

    // We must have at least one more free byte after the prefixes, which will be either the opcode, either
    // XOP/VEX/EVEX/MVEX prefix.
    RET_GT((ND_SIZET)Offset + 1, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // Try to match a XOP/VEX/EVEX/MVEX prefix.
    if (ND_PREF_CODE_EX == gPrefixesMap[Code[Offset]])
    {
        // Check for XOP
        if (Code[Offset] == ND_PREFIX_XOP)
        {
            status = NdFetchXop(Instrux, Code, Offset, Size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }
        else if (Code[Offset] == ND_PREFIX_VEX_2B)
        {
            status = NdFetchVex2(Instrux, Code, Offset, Size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }
        else if (Code[Offset] == ND_PREFIX_VEX_3B)
        {
            status = NdFetchVex3(Instrux, Code, Offset, Size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }
        else if (Code[Offset] == ND_PREFIX_EVEX)
        {
            status = NdFetchEvex(Instrux, Code, Offset, Size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }
        else if (Code[Offset] == ND_PREFIX_REX2)
        {
            status = NdFetchRex2(Instrux, Code, Offset, Size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }
        else
        {
            return ND_STATUS_INVALID_INSTRUX;
        }
    }

done_prefixes:
    // The total length of the instruction is the total length of the prefixes right now.
    Instrux->PrefLength = Instrux->OpOffset = Instrux->Length;

    return ND_STATUS_SUCCESS;
}


//
// NdFetchOpcode
//
static NDSTATUS
NdFetchOpcode(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // At least one byte must be available, for the fetched opcode.
    RET_GT((ND_SIZET)Offset + 1, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // With REX2, only legacy map & 0x0F map are valid. A single opcode byte can be present, and no
    // opcode extensions are accepted (for example, 0x0F 0x38 is invalid).
    if (Instrux->HasRex2 && Instrux->OpLength != 0)
    {
        return ND_STATUS_INVALID_ENCODING;
    }

    Instrux->OpCodeBytes[Instrux->OpLength++] = Code[Offset];

    Instrux->Length++;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchModrm
//
static NDSTATUS
NdFetchModrm(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // At least one byte must be available, for the modrm byte.
    RET_GT((ND_SIZET)Offset + 1, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // If we get called, we assume we have ModRM.
    Instrux->HasModRm = ND_TRUE;

    // Fetch the ModRM byte & update the offset and the instruction length.
    Instrux->ModRm.ModRm = Code[Offset];
    Instrux->ModRmOffset = Offset;

    Instrux->Length++, Offset++;

    // Make sure we don't exceed the maximum instruction length.
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchModrmAndSib
//
static NDSTATUS
NdFetchModrmAndSib(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    // At least one byte must be available, for the modrm byte.
    RET_GT((ND_SIZET)Offset + 1, Size, ND_STATUS_BUFFER_TOO_SMALL);

    // If we get called, we assume we have ModRM.
    Instrux->HasModRm = ND_TRUE;

    // Fetch the ModRM byte & update the offset and the instruction length.
    Instrux->ModRm.ModRm = Code[Offset];
    Instrux->ModRmOffset = Offset;

    Instrux->Length++, Offset++;

    // Make sure we don't exceed the maximum instruction length.
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    // If needed, fetch the SIB.
    if ((Instrux->ModRm.rm == NDR_RSP) && (Instrux->ModRm.mod != 3) && (Instrux->AddrMode != ND_ADDR_16))
    {
        // At least one more byte must be available, for the sib.
        RET_GT((ND_SIZET)Offset + 1, Size, ND_STATUS_BUFFER_TOO_SMALL);

        // SIB present.
        Instrux->HasSib = ND_TRUE;

        Instrux->Sib.Sib = Code[Offset];
        Instrux->Length++;

        // Make sure we don't exceed the maximum instruction length.
        if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
        {
            return ND_STATUS_INSTRUCTION_TOO_LONG;
        }
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchDisplacement
//
static NDSTATUS
NdFetchDisplacement(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
//
// Will decode the displacement from the instruction. Will fill in extracted information in Instrux,
// and will update the instruction length.
//
{
    ND_UINT8 displSize;

    displSize = 0;

    if (ND_ADDR_16 == Instrux->AddrMode)
    {
        displSize = gDispsizemap16[Instrux->ModRm.mod][Instrux->ModRm.rm];
    }
    else
    {
        displSize = gDispsizemap[Instrux->ModRm.mod][Instrux->HasSib ? Instrux->Sib.base : Instrux->ModRm.rm];
    }

    if (0 != displSize)
    {
        // Make sure enough buffer space is available.
        RET_GT((ND_SIZET)Offset + displSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

        // If we get here, we have displacement.
        Instrux->HasDisp = ND_TRUE;

        Instrux->Displacement = (ND_UINT32)NdFetchData(Code + Offset, displSize);

        // Fill in displacement info.
        Instrux->DispLength = displSize;
        Instrux->DispOffset = Offset;
        Instrux->Length += displSize;
        if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
        {
            return ND_STATUS_INSTRUCTION_TOO_LONG;
        }
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchModrmSibDisplacement
//
static NDSTATUS
NdFetchModrmSibDisplacement(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size
    )
{
    NDSTATUS status;

    status = NdFetchModrmAndSib(Instrux, Code, Offset, Size);
    if (!ND_SUCCESS(status))
    {
        return status;
    }

    return NdFetchDisplacement(Instrux, Code, Instrux->Length, Size);
}


//
// NdFetchAddressFar
//
static NDSTATUS
NdFetchAddressFar(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT8 AddressSize
    )
{
    RET_GT((ND_SIZET)Offset + AddressSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

    Instrux->HasAddr = ND_TRUE;
    Instrux->AddrLength = AddressSize;
    Instrux->AddrOffset = Offset;

    Instrux->Address.Ip = (ND_UINT32)NdFetchData(Code + Offset, Instrux->AddrLength - 2);
    Instrux->Address.Cs = (ND_UINT16)NdFetchData(Code + Offset + Instrux->AddrLength - 2, 2);

    Instrux->Length += Instrux->AddrLength;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchAddressNear
//
static NDSTATUS
NdFetchAddressNear(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT8 AddressSize
    )
{
    RET_GT((ND_SIZET)Offset + AddressSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

    Instrux->HasAddrNear = ND_TRUE;
    Instrux->AddrLength = AddressSize;
    Instrux->AddrOffset = Offset;

    Instrux->AddressNear = (ND_UINT64)NdFetchData(Code + Offset, Instrux->AddrLength);

    Instrux->Length += Instrux->AddrLength;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchImmediate
//
static NDSTATUS
NdFetchImmediate(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT8 ImmediateSize
    )
{
    ND_UINT64 imm;

    RET_GT((ND_SIZET)Offset + ImmediateSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

    imm = NdFetchData(Code + Offset, ImmediateSize);

    if (Instrux->HasImm2)
    {
        return ND_STATUS_INVALID_INSTRUX;
    }
    else if (Instrux->HasImm1)
    {
        Instrux->HasImm2 = ND_TRUE;
        Instrux->Imm2Length = ImmediateSize;
        Instrux->Imm2Offset = Offset;
        Instrux->Immediate2 = (ND_UINT8)imm;
    }
    else
    {
        Instrux->HasImm1 = ND_TRUE;
        Instrux->Imm1Length = ImmediateSize;
        Instrux->Imm1Offset = Offset;
        Instrux->Immediate1 = imm;
    }

    Instrux->Length += ImmediateSize;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchRelativeOffset
//
static NDSTATUS
NdFetchRelativeOffset(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT8 RelOffsetSize
    )
{
    // Make sure we don't outrun the buffer.
    RET_GT((ND_SIZET)Offset + RelOffsetSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

    Instrux->HasRelOffs = ND_TRUE;
    Instrux->RelOffsLength = RelOffsetSize;
    Instrux->RelOffsOffset = Offset;

    Instrux->RelativeOffset = (ND_UINT32)NdFetchData(Code + Offset, RelOffsetSize);

    Instrux->Length += RelOffsetSize;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchMoffset
//
static NDSTATUS
NdFetchMoffset(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT8 MoffsetSize
    )
{
    RET_GT((ND_SIZET)Offset + MoffsetSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

    Instrux->HasMoffset = ND_TRUE;
    Instrux->MoffsetLength = MoffsetSize;
    Instrux->MoffsetOffset = Offset;

    Instrux->Moffset = NdFetchData(Code + Offset, MoffsetSize);

    Instrux->Length += MoffsetSize;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdFetchSseImmediate
//
static NDSTATUS
NdFetchSseImmediate(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT8 SseImmSize
    )
{
    RET_GT((ND_SIZET)Offset + SseImmSize, Size, ND_STATUS_BUFFER_TOO_SMALL);

    Instrux->HasSseImm = ND_TRUE;
    Instrux->SseImmOffset = Offset;
    Instrux->SseImmediate = *(Code + Offset);

    Instrux->Length += SseImmSize;
    if (Instrux->Length > ND_MAX_INSTRUCTION_LENGTH)
    {
        return ND_STATUS_INSTRUCTION_TOO_LONG;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdGetSegOverride
//
static ND_UINT8
NdGetSegOverride(
    INSTRUX *Instrux,
    ND_UINT8 DefaultSeg
    )
{
    // Return default seg, if no override present.
    if (Instrux->Seg == 0)
    {
        return DefaultSeg;
    }

    // In 64 bit mode, the segment override is ignored, except for FS and GS.
    if ((Instrux->DefCode == ND_CODE_64) &&
        (Instrux->Seg != ND_PREFIX_G2_SEG_FS) &&
        (Instrux->Seg != ND_PREFIX_G2_SEG_GS))
    {
        return DefaultSeg;
    }

    switch (Instrux->Seg)
    {
    case ND_PREFIX_G2_SEG_CS:
        return NDR_CS;
    case ND_PREFIX_G2_SEG_DS:
        return NDR_DS;
    case ND_PREFIX_G2_SEG_ES:
        return NDR_ES;
    case ND_PREFIX_G2_SEG_SS:
        return NDR_SS;
    case ND_PREFIX_G2_SEG_FS:
        return NDR_FS;
    case ND_PREFIX_G2_SEG_GS:
        return NDR_GS;
    default:
        return DefaultSeg;
    }
}


//
// NdGetCompDispSize
//
static ND_UINT8
NdGetCompDispSize(
    const INSTRUX *Instrux,
    ND_UINT32 MemSize
    )
{
    static const ND_UINT8 fvszLut[4] = { 16, 32, 64, 0 };
    static const ND_UINT8 hvszLut[4] = { 8, 16, 32, 0 };
    static const ND_UINT8 qvszLut[4] = { 4, 8, 16, 0 };
    static const ND_UINT8 dupszLut[4] = { 8, 32, 64, 0 };
    static const ND_UINT8 fvmszLut[4] = { 16, 32, 64, 0 };
    static const ND_UINT8 hvmszLut[4] = { 8, 16, 32, 0 };
    static const ND_UINT8 qvmszLut[4] = { 4, 8, 16, 0 };
    static const ND_UINT8 ovmszLut[4] = { 2, 4, 8, 0 };

    if (Instrux->HasBroadcast)
    {
        // If the instruction uses broadcast, then compressed displacement will use the size of the element as scale:
        // - 2 when broadcasting 16 bit
        // - 4 when broadcasting 32 bit
        // - 8 when broadcasting 64 bit
        return (ND_UINT8)MemSize;
    }

    switch (Instrux->TupleType)
    {
    case ND_TUPLE_FV:
        return fvszLut[Instrux->Exs.l];
    case ND_TUPLE_HV:
        return hvszLut[Instrux->Exs.l];
    case ND_TUPLE_QV:
        return qvszLut[Instrux->Exs.l];
    case ND_TUPLE_DUP:
        return dupszLut[Instrux->Exs.l];
    case ND_TUPLE_FVM:
        return fvmszLut[Instrux->Exs.l];
    case ND_TUPLE_HVM:
        return hvmszLut[Instrux->Exs.l];
    case ND_TUPLE_QVM:
        return qvmszLut[Instrux->Exs.l];
    case ND_TUPLE_OVM:
        return ovmszLut[Instrux->Exs.l];
    case ND_TUPLE_M128:
        return 16;
    case ND_TUPLE_T1S8:
        return 1;
    case ND_TUPLE_T1S16:
        return 2;
    case ND_TUPLE_T1S:
        return !!(Instrux->Attributes & ND_FLAG_WIG) ? 4 : Instrux->Exs.w ? 8 : 4;
    case ND_TUPLE_T1F:
        return (ND_UINT8)MemSize;
    case ND_TUPLE_T2:
        return Instrux->Exs.w ? 16 : 8;
    case ND_TUPLE_T4:
        return Instrux->Exs.w ? 32 : 16;
    case ND_TUPLE_T8:
        return 32;
    case ND_TUPLE_T1_4X:
        return 16;
    default:
        // Default - we assume byte granularity for memory accesses, therefore, no scaling will be done.
        return 1;
    }
}


//
// NdParseMemoryOperand16
//
static NDSTATUS
NdParseMemoryOperand16(
    INSTRUX *Instrux,
    ND_OPERAND *Operand
    )
{
    if (Instrux->Attributes & ND_FLAG_NOA16)
    {
        return ND_STATUS_16_BIT_ADDRESSING_NOT_SUPPORTED;
    }

    switch (Instrux->ModRm.rm)
    {
    case 0:
        // [bx + si]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.HasIndex = ND_TRUE;
        Operand->Info.Memory.Scale = 1;
        Operand->Info.Memory.Base = NDR_BX;
        Operand->Info.Memory.Index = NDR_SI;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.IndexSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_DS;
        break;
    case 1:
        // [bx + di]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.HasIndex = ND_TRUE;
        Operand->Info.Memory.Scale = 1;
        Operand->Info.Memory.Base = NDR_BX;
        Operand->Info.Memory.Index = NDR_DI;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.IndexSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_DS;
        break;
    case 2:
        // [bp + si]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.HasIndex = ND_TRUE;
        Operand->Info.Memory.Scale = 1;
        Operand->Info.Memory.Base = NDR_BP;
        Operand->Info.Memory.Index = NDR_SI;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.IndexSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_SS;
        break;
    case 3:
        // [bp + di]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.HasIndex = ND_TRUE;
        Operand->Info.Memory.Scale = 1;
        Operand->Info.Memory.Base = NDR_BP;
        Operand->Info.Memory.Index = NDR_DI;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.IndexSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_SS;
        break;
    case 4:
        // [si]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.Base = NDR_SI;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_DS;
        break;
    case 5:
        // [di]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.Base = NDR_DI;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_DS;
        break;
    case 6:
        // [bp]
        if (Instrux->ModRm.mod != 0)
        {
            // If mod is not zero, than we have "[bp + displacement]".
            Operand->Info.Memory.HasBase = ND_TRUE;
            Operand->Info.Memory.Base = NDR_BP;
            Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
            Operand->Info.Memory.Seg = NDR_SS;
        }
        else
        {
            // If mod is zero, than we only have a displacement that is used to directly address mem.
            Operand->Info.Memory.Seg = NDR_DS;
        }
        break;
    case 7:
        // [bx]
        Operand->Info.Memory.HasBase = ND_TRUE;
        Operand->Info.Memory.Base = NDR_BX;
        Operand->Info.Memory.BaseSize = ND_SIZE_16BIT;
        Operand->Info.Memory.Seg = NDR_DS;
        break;
    }

    // Store the displacement.
    Operand->Info.Memory.HasDisp = !!Instrux->HasDisp;
    Operand->Info.Memory.DispSize = Instrux->DispLength;
    Operand->Info.Memory.Disp = Instrux->HasDisp ? ND_SIGN_EX(Instrux->DispLength, Instrux->Displacement) : 0;

    return ND_STATUS_SUCCESS;
}


//
// NdParseMemoryOperand3264
//
static NDSTATUS
NdParseMemoryOperand3264(
    INSTRUX *Instrux,
    ND_OPERAND *Operand,
    ND_REG_SIZE VsibRegSize
    )
{
    ND_UINT8 defsize = (Instrux->AddrMode == ND_ADDR_32 ? ND_SIZE_32BIT : ND_SIZE_64BIT);

    // Implicit segment is DS.
    Operand->Info.Memory.Seg = NDR_DS;

    if (Instrux->HasSib)
    {
        // Check for base.
        if ((Instrux->ModRm.mod == 0) && (Instrux->Sib.base == NDR_RBP))
        {
            // Mod is mem without displacement and base reg is RBP -> no base reg used.
            // Note that this addressing mode is not RIP relative.
        }
        else
        {
            Operand->Info.Memory.HasBase = ND_TRUE;
            Operand->Info.Memory.BaseSize = defsize;
            Operand->Info.Memory.Base = (ND_UINT8)(Instrux->Exs.b << 3) | Instrux->Sib.base;

            if (Instrux->Exs.b4 != 0)
            {
                // If APX is present, extend the base.
                if (Instrux->FeatMode & ND_FEAT_APX)
                {
                    Operand->Info.Memory.Base |= Instrux->Exs.b4 << 4;
                }
                else
                {
                    return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
                }
            }

            if ((Operand->Info.Memory.Base == NDR_RSP) || (Operand->Info.Memory.Base == NDR_RBP))
            {
                Operand->Info.Memory.Seg = NDR_SS;
            }
        }

        // Check for index.
        if (ND_HAS_VSIB(Instrux))
        {
            // With VSIB, the index reg can be 4 (RSP equivalent). Bit 4 of the 32-bit index register is given by the
            // EVEX.V' field.
            Operand->Info.Memory.HasIndex = ND_TRUE;
            Operand->Info.Memory.IndexSize = defsize;
            Operand->Info.Memory.Index = (ND_UINT8)((Instrux->Exs.vp << 4) | (Instrux->Exs.x << 3) | Instrux->Sib.index);
            Operand->Info.Memory.IndexSize = (ND_UINT8)VsibRegSize;
            Operand->Info.Memory.Scale = 1 << Instrux->Sib.scale;
        }
        else
        {
            // Regular SIB, index RSP is ignored. Bit 4 of the 32-bit index register is given by the X4 field.
            Operand->Info.Memory.Index = (ND_UINT8)(Instrux->Exs.x << 3) | Instrux->Sib.index;

            if (Instrux->Exs.x4 != 0)
            {
                // If APX is present, extend the index.
                if (Instrux->FeatMode & ND_FEAT_APX)
                {
                    Operand->Info.Memory.Index |= Instrux->Exs.x4 << 4;
                }
                else
                {
                    return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
                }
            }

            if (Operand->Info.Memory.Index != NDR_RSP)
            {
                // Index * Scale is present.
                Operand->Info.Memory.HasIndex = ND_TRUE;
                Operand->Info.Memory.IndexSize = defsize;
                Operand->Info.Memory.Scale = 1 << Instrux->Sib.scale;
            }
        }
    }
    else
    {
        if ((Instrux->ModRm.mod == 0) && (Instrux->ModRm.rm == NDR_RBP))
        {
            //
            // RIP relative addressing addresses a memory region relative to the current RIP; However,
            // the current RIP, when executing the current instruction, is already updated and points
            // to the next instruction, therefore, we must add the instruction length also to the final
            // address. Note that RIP relative addressing is used even if the instruction uses 32 bit
            // addressing, as long as we're in long mode.
            //
            Operand->Info.Memory.IsRipRel = Instrux->IsRipRelative = (Instrux->DefCode == ND_CODE_64);

            // Some instructions (example: MPX) don't support RIP relative addressing.
            if (Operand->Info.Memory.IsRipRel && !!(Instrux->Attributes & ND_FLAG_NO_RIP_REL))
            {
                return ND_STATUS_RIP_REL_ADDRESSING_NOT_SUPPORTED;
            }
        }
        else
        {
            Operand->Info.Memory.HasBase = ND_TRUE;
            Operand->Info.Memory.BaseSize = defsize;
            Operand->Info.Memory.Base = (ND_UINT8)(Instrux->Exs.b << 3) | Instrux->ModRm.rm;

            if (Instrux->Exs.b4 != 0)
            {
                // If APX is present, extend the base register.
                if (Instrux->FeatMode & ND_FEAT_APX)
                {
                    Operand->Info.Memory.Base |= Instrux->Exs.b4 << 4;
                }
                else
                {
                    return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
                }
            }

            if ((Operand->Info.Memory.Base == NDR_RSP) || (Operand->Info.Memory.Base == NDR_RBP))
            {
                Operand->Info.Memory.Seg = NDR_SS;
            }
        }
    }

    Operand->Info.Memory.HasDisp = Instrux->HasDisp;
    Operand->Info.Memory.DispSize = Instrux->DispLength;
    Operand->Info.Memory.Disp = Instrux->HasDisp ? ND_SIGN_EX(Instrux->DispLength, Instrux->Displacement) : 0;

    return ND_STATUS_SUCCESS;
}



//
// NdParseOperand
//
static NDSTATUS
NdParseOperand(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 Offset,
    ND_SIZET Size,
    ND_UINT32 Index,
    ND_UINT64 Specifier
    )
{
    NDSTATUS status;
    PND_OPERAND operand;
    ND_UINT8 opt, ops, opf, opa, opd, opb;
    ND_REG_SIZE vsibRegSize;
    ND_UINT8 vsibIndexSize, vsibIndexCount;
    ND_OPERAND_SIZE size;
    ND_BOOL width;

    // pre-init
    status = ND_STATUS_SUCCESS;
    vsibRegSize = 0;
    vsibIndexSize = vsibIndexCount = 0;
    size = 0;

    // Get actual width.
    width = Instrux->Exs.w && !(Instrux->Attributes & ND_FLAG_WIG);

    // Get operand components.
    opt = ND_OP_TYPE(Specifier);
    ops = ND_OP_SIZE(Specifier);
    opf = ND_OP_FLAGS(Specifier);
    opa = ND_OP_ACCESS(Specifier);
    opd = ND_OP_DECORATORS(Specifier);
    opb = ND_OP_BLOCK(Specifier);

    // Get a pointer to our op.
    operand = &Instrux->Operands[Index];

    // Fill in the flags.
    operand->Flags.Flags = opf;

    // Store operand access modes.
    operand->Access.Access = opa;

    // Implicit operand access, by default.
    operand->Encoding = ND_OPE_S;


    //
    // Fill in operand size.
    //
    switch (ops)
    {
    case ND_OPS_asz:
        // Size given by the address mode.
        size = 2 << Instrux->AddrMode;
        break;

    case ND_OPS_ssz:
        // Size given by the stack mode.
        size = 2 << Instrux->DefStack;
        break;

    case ND_OPS_0:
        // No memory access. 0 operand size.
        size = 0;
        break;

    case ND_OPS_b:
        // 8 bits.
        size = ND_SIZE_8BIT;
        break;

    case ND_OPS_w:
        // 16 bits.
        size = ND_SIZE_16BIT;
        break;

    case ND_OPS_d:
        // 32 bits.
        size = ND_SIZE_32BIT;
        break;

    case ND_OPS_q:
        // 64 bits.
        size = ND_SIZE_64BIT;
        break;

    case ND_OPS_dq:
        // 128 bits. 
        size = ND_SIZE_128BIT;
        break;

    case ND_OPS_qq:
        // 256 bits.
        size = ND_SIZE_256BIT;
        break;

    case ND_OPS_oq:
        // 512 bits.
        size = ND_SIZE_512BIT;
        break;

    case ND_OPS_fa:
        // 80 bits packed BCD.
        size = ND_SIZE_80BIT;
        break;

    case ND_OPS_fw:
        // 16 bits real number.
        size = ND_SIZE_16BIT;
        break;

    case ND_OPS_fd:
        // 32 bits real number.
        size = ND_SIZE_32BIT;
        break;

    case ND_OPS_fq:
        // 64 bits real number.
        size = ND_SIZE_64BIT;
        break;

    case ND_OPS_ft:
        // 80 bits real number.
        size = ND_SIZE_80BIT;
        break;

    case ND_OPS_fe:
        // 14 bytes or 28 bytes FPU environment.
        size = (Instrux->EfOpMode == ND_OPSZ_16) ? ND_SIZE_112BIT : ND_SIZE_224BIT;
        break;

    case ND_OPS_fs:
        // 94 bytes or 108 bytes FPU state.
        size = (Instrux->EfOpMode == ND_OPSZ_16) ? ND_SIZE_752BIT : ND_SIZE_864BIT;
        break;

    case ND_OPS_rx:
        // 512 bytes extended state.
        size = ND_SIZE_4096BIT;
        break;

    case ND_OPS_cl:
        // The size of one cache line.
        size = ND_SIZE_CACHE_LINE;
        break;

    case ND_OPS_v:
        // 16, 32 or 64 bits.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };

            size = szLut[Instrux->EfOpMode];
        }
        break;

    case ND_OPS_y:
        // 64 bits (64-bit opsize), 32 bits othwerwise.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_32BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };

            size = szLut[Instrux->EfOpMode];
        }
        break;

    case ND_OPS_yf:
        // 64 bits (64-bit mode), 32 bits (16, 32-bit opsize).
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_32BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };

            size = szLut[Instrux->DefCode];
        }
        break;

    case ND_OPS_z:
        // 16 bits (16-bit opsize) or 32 bits (32 or 64-bit opsize).
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_32BIT };

            size = szLut[Instrux->EfOpMode];
        }
        break;

    case ND_OPS_a:
        // 2 x 16 bits (16-bit opsize) or 2 x 32 bits (32-bit opsize).
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT * 2, ND_SIZE_32BIT * 2, 0 };

            if (Instrux->DefCode > ND_CODE_32)
            {
                return ND_STATUS_INVALID_INSTRUX;
            }

            size = szLut[Instrux->EfOpMode];
        }
        break;

    case ND_OPS_c:
        // 8 bits (16-bit opsize) or 16 bits (32-bit opsize).
        switch (Instrux->DefCode)
        {
        case ND_CODE_16:
            size = Instrux->HasOpSize ? ND_SIZE_16BIT : ND_SIZE_8BIT;
            break;
        case ND_CODE_32:
            size = Instrux->HasOpSize ? ND_SIZE_16BIT : ND_SIZE_32BIT;
            break;
        case ND_CODE_64:
            size = ND_SIZE_64BIT;
            break;
        default:
            return ND_STATUS_INVALID_INSTRUX;
        }
        break;

    case ND_OPS_p:
        // 32, 48 or 80 bits pointer.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_32BIT, ND_SIZE_48BIT, ND_SIZE_80BIT };

            size = szLut[Instrux->EfOpMode];
        }
        break;

    case ND_OPS_s:
        // 48 or 80 bits descriptor.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_48BIT, ND_SIZE_48BIT, ND_SIZE_80BIT };

            size = szLut[Instrux->DefCode];
        }
        break;

    case ND_OPS_l:
        // 64 (16 or 32-bit opsize) or 128 bits (64-bit opsize).
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_64BIT, ND_SIZE_64BIT, ND_SIZE_128BIT };

            size = szLut[Instrux->DefCode];
        }
        break;

    case ND_OPS_x:
        // lower vector = 128 (128-bit vlen) or 256 bits (256-bit vlen).
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_128BIT, ND_SIZE_256BIT, ND_SIZE_512BIT };

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_fv:
        // full vector = 128, 256 or 512 bits.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_128BIT, ND_SIZE_256BIT, ND_SIZE_512BIT };

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_uv:
        // upper vector = 256 bits (256-bit vlen) or 512 bits (512-bit vlen)
        {
            static const ND_UINT8 szLut[3] = { 0, ND_SIZE_256BIT, ND_SIZE_512BIT };

            if (ND_VECM_128 == Instrux->EfVecMode)
            {
                return ND_STATUS_INVALID_INSTRUX;
            }

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_ev:
        // eighth vector = 16, 32 or 64 bits.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_qv:
        // quarter vector = 32, 64 or 128 bits.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_32BIT, ND_SIZE_64BIT, ND_SIZE_128BIT };

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_hv:
        // half vector = 64, 128 or 256 bits.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_64BIT, ND_SIZE_128BIT, ND_SIZE_256BIT };

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_pd:
    case ND_OPS_ps:
    case ND_OPS_ph:
        // 128 or 256 bits.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_128BIT, ND_SIZE_256BIT, ND_SIZE_512BIT };

            size = szLut[Instrux->EfVecMode];
        }
        break;

    case ND_OPS_sd:
        // 128 bits scalar element (double precision).
        size = ND_SIZE_64BIT;
        break;

    case ND_OPS_ss:
        // 128 bits scalar element (single precision).
        size = ND_SIZE_32BIT;
        break;

    case ND_OPS_sh:
        // FP16 Scalar element.
        size = ND_SIZE_16BIT;
        break;

    case ND_OPS_mib:
        // MIB addressing, the base & the index are used to form a pointer.
        size = 0;
        break;

    case ND_OPS_vm32x:
    case ND_OPS_vm32y:
    case ND_OPS_vm32z:
        // 32 bit indexes from XMM, YMM or ZMM register.
        vsibIndexSize  = ND_SIZE_32BIT;
        vsibIndexCount = (Instrux->Exs.l == 0) ? 4 : ((Instrux->Exs.l == 1) ? 8 : 16);
        vsibRegSize = (ops == ND_OPS_vm32x) ? ND_SIZE_128BIT :
                      (ops == ND_OPS_vm32y) ? ND_SIZE_256BIT :
                                              ND_SIZE_512BIT;
        size = vsibIndexCount * (width ? ND_SIZE_64BIT : ND_SIZE_32BIT);
        break;

    case ND_OPS_vm32h:
        // 32 bit indexes from XMM or YMM.
        vsibIndexSize = ND_SIZE_32BIT;
        vsibIndexCount = (Instrux->Exs.l == 0) ? 2 : ((Instrux->Exs.l == 1) ? 4 : 8);
        vsibRegSize = (Instrux->Exs.l == 0) ? ND_SIZE_128BIT :
                      (Instrux->Exs.l == 1) ? ND_SIZE_128BIT :
                                              ND_SIZE_256BIT;
        size = vsibIndexCount * (width ? ND_SIZE_64BIT : ND_SIZE_32BIT);
        break;

    case ND_OPS_vm32n:
        // 32 bit indexes from XMM, YMM or ZMM register.
        vsibIndexSize = ND_SIZE_32BIT;
        vsibIndexCount = (Instrux->Exs.l == 0) ? 4 : ((Instrux->Exs.l == 1) ? 8 : 16);
        vsibRegSize = (Instrux->Exs.l == 0) ? ND_SIZE_128BIT :
                      (Instrux->Exs.l == 1) ? ND_SIZE_256BIT :
                                              ND_SIZE_512BIT;
        size = vsibIndexCount * (width ? ND_SIZE_64BIT : ND_SIZE_32BIT);
        break;

    case ND_OPS_vm64x:
    case ND_OPS_vm64y:
    case ND_OPS_vm64z:
        // 64 bit indexes from XMM, YMM or ZMM register.
        vsibIndexSize = ND_SIZE_64BIT;
        vsibIndexCount = (Instrux->Exs.l == 0) ? 2 : ((Instrux->Exs.l == 1) ? 4 : 8);
        vsibRegSize = (ops == ND_OPS_vm64x) ? ND_SIZE_128BIT :
                      (ops == ND_OPS_vm64y) ? ND_SIZE_256BIT :
                                              ND_SIZE_512BIT;
        size = vsibIndexCount * (width ? ND_SIZE_64BIT : ND_SIZE_32BIT);
        break;

    case ND_OPS_vm64h:
        // 64 bit indexes from XMM or YMM.
        vsibIndexSize = ND_SIZE_64BIT;
        vsibIndexCount = (Instrux->Exs.l == 0) ? 1 : ((Instrux->Exs.l == 1) ? 2 : 4);
        vsibRegSize = (Instrux->Exs.l == 0) ? ND_SIZE_128BIT :
                      (Instrux->Exs.l == 1) ? ND_SIZE_128BIT :
                                              ND_SIZE_256BIT;
        size = vsibIndexCount * (width ? ND_SIZE_64BIT : ND_SIZE_32BIT);
        break;

    case ND_OPS_vm64n:
        // 64 bit indexes from XMM, YMM or ZMM register.
        vsibIndexSize = ND_SIZE_64BIT;
        vsibIndexCount = (Instrux->Exs.l == 0) ? 2 : ((Instrux->Exs.l == 1) ? 4 : 8);
        vsibRegSize = (Instrux->Exs.l == 0) ? ND_SIZE_128BIT :
                      (Instrux->Exs.l == 1) ? ND_SIZE_256BIT :
                                              ND_SIZE_512BIT;
        size = vsibIndexCount * (width ? ND_SIZE_64BIT : ND_SIZE_32BIT);
        break;

    case ND_OPS_v2:
    case ND_OPS_v3:
    case ND_OPS_v4:
    case ND_OPS_v5:
    case ND_OPS_v8:
        // Multiple words accessed.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };
            ND_UINT8 scale = 1;

            scale = (ops == ND_OPS_v2) ? 2 : 
                    (ops == ND_OPS_v3) ? 3 : 
                    (ops == ND_OPS_v4) ? 4 : 
                    (ops == ND_OPS_v5) ? 5 : 8;

            size =  scale * szLut[Instrux->EfOpMode];
        }
        break;

    case ND_OPS_12:
        // SAVPREVSSP instruction reads/writes 4 + 8 bytes from the shadow stack.
        size = 12;
        break;

    case ND_OPS_t:
        // Tile register. The actual size depends on how the TILECFG register has been programmed, but it can be 
        // up to 1K in size.
        size = ND_SIZE_1KB;
        break;

    case ND_OPS_384:
        // 384 bit Key Locker handle.
        size = ND_SIZE_384BIT;
        break;

    case ND_OPS_512:
        // 512 bit Key Locker handle.
        size = ND_SIZE_512BIT;
        break;

    case ND_OPS_4096:
        // 64 entries x 64 bit per entry = 4096 bit MSR address/value list.
        size = ND_SIZE_4096BIT;
        break;

    case ND_OPS_unknown:
        size = ND_SIZE_UNKNOWN;
        break;

    default:
        return ND_STATUS_INVALID_INSTRUX;
    }

    // Store operand info.
    operand->Size = size;

    //
    // Fill in the operand type.
    //
    switch (opt)
    {
    case ND_OPT_1:
        // operand is an implicit constant (used by shift/rotate instruction).
        operand->Type = ND_OP_CONST;
        operand->Encoding = ND_OPE_1;
        operand->Info.Constant.Const = 1;
        break;

    case ND_OPT_rIP:
        // The operand is the instruction pointer.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_RIP;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = 0;
        Instrux->RipAccess |= operand->Access.Access;
        // Fill in branch information.
        Instrux->BranchInfo.IsBranch = 1;
        Instrux->BranchInfo.IsConditional = Instrux->Category == ND_CAT_COND_BR;
        // Indirect branches are those which get their target address from a register or memory, including RET family.
        Instrux->BranchInfo.IsIndirect = ((!Instrux->Operands[0].Flags.IsDefault) && 
            ((Instrux->Operands[0].Type == ND_OP_REG) || (Instrux->Operands[0].Type == ND_OP_MEM))) || 
            (Instrux->Category == ND_CAT_RET);
        // CS operand is ALWAYS before rIP.
        Instrux->BranchInfo.IsFar = !!(Instrux->CsAccess & ND_ACCESS_ANY_WRITE);
        break;

    case ND_OPT_rAX:
        // Operand is the accumulator.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RAX;
        break;

    case ND_OPT_AH:
        // Operand is the accumulator.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = ND_SIZE_8BIT;
        operand->Info.Register.Reg = NDR_AH;
        operand->Info.Register.IsHigh8 = ND_TRUE;
        break;

    case ND_OPT_rCX:
        // Operand is the counter register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RCX;
        break;

    case ND_OPT_rDX:
        // Operand is rDX.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RDX;
        break;

    case ND_OPT_rBX:
        // Operand is BX.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RBX;
        break;

    case ND_OPT_rBP:
        // Operand is rBP.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RBP;
        break;

    case ND_OPT_rSP:
        // Operand is rSP.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RSP;
        break;

    case ND_OPT_rSI:
        // Operand is rSI.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RSI;
        break;

    case ND_OPT_rDI:
        // Operand is rDI.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_RDI;
        break;

    case ND_OPT_rR8:
        // Operand is R8.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_R8;
        break;

    case ND_OPT_rR9:
        // Operand is R9.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_R9;
        break;

    case ND_OPT_rR11:
        // Operand is R11.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_R11;
        break;

    case ND_OPT_CS:
        // Operand is the CS register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_CS;
        Instrux->CsAccess |= operand->Access.Access;
        break;

    case ND_OPT_SS:
        // Operand is the SS register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_SS;
        break;

    case ND_OPT_DS:
        // Operand is the DS register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_DS;
        break;

    case ND_OPT_ES:
        // Operand is the ES register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_ES;
        break;

    case ND_OPT_FS:
        // Operand is the FS register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_FS;
        break;

    case ND_OPT_GS:
        // Operand is the GS register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_GS;
        break;

    case ND_OPT_ST0:
        // Operand is the ST(0) register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_FPU;
        operand->Info.Register.Size = ND_SIZE_80BIT;
        operand->Info.Register.Reg = 0;
        break;

    case ND_OPT_STi:
        // Operand is the ST(i) register.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_FPU;
        operand->Info.Register.Size = ND_SIZE_80BIT;
        operand->Info.Register.Reg = Instrux->ModRm.rm;
        break;

    case ND_OPT_XMM0:
    case ND_OPT_XMM1:
    case ND_OPT_XMM2:
    case ND_OPT_XMM3:
    case ND_OPT_XMM4:
    case ND_OPT_XMM5:
    case ND_OPT_XMM6:
    case ND_OPT_XMM7:
        // Operand is a hard-coded XMM register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SSE;
        operand->Info.Register.Size = ND_SIZE_128BIT;
        operand->Info.Register.Reg = opt - ND_OPT_XMM0;
        break;

    // Special operands. These are always implicit, and can't be encoded inside the instruction.
    case ND_OPT_CR0:
        // The operand is implicit and is control register 0.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_CR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_CR0;
        break;

    case ND_OPT_GDTR:
        // The operand is implicit and is the global descriptor table register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_GDTR;
        break;

    case ND_OPT_IDTR:
        // The operand is implicit and is the interrupt descriptor table register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_IDTR;
        break;

    case ND_OPT_LDTR:
        // The operand is implicit and is the local descriptor table register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_LDTR;
        break;

    case ND_OPT_TR:
        // The operand is implicit and is the task register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = NDR_TR;
        break;

    case ND_OPT_X87CONTROL:
        // The operand is implicit and is the x87 control word.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = ND_SIZE_16BIT;
        operand->Info.Register.Reg = NDR_X87_CONTROL;
        break;

    case ND_OPT_X87TAG:
        // The operand is implicit and is the x87 tag word.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = ND_SIZE_16BIT;
        operand->Info.Register.Reg = NDR_X87_TAG;
        break;

    case ND_OPT_X87STATUS:
        // The operand is implicit and is the x87 status word.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SYS;
        operand->Info.Register.Size = ND_SIZE_16BIT;
        operand->Info.Register.Reg = NDR_X87_STATUS;
        break;

    case ND_OPT_MXCSR:
        // The operand is implicit and is the MXCSR.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MXCSR;
        operand->Info.Register.Size = ND_SIZE_32BIT;
        operand->Info.Register.Reg = 0;
        break;

    case ND_OPT_PKRU:
        // The operand is the PKRU register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_PKRU;
        operand->Info.Register.Size = ND_SIZE_32BIT;
        operand->Info.Register.Reg = 0;
        break;

    case ND_OPT_SSP:
        // The operand is the SSP register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_SSP;
        operand->Info.Register.Size = operand->Size;
        operand->Info.Register.Reg = 0;
        break;

    case ND_OPT_UIF:
        // The operand is the User Interrupt Flag.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_UIF;
        operand->Info.Register.Size = ND_SIZE_8BIT; // 1 bit, in fact, but there is no size defined for one bit.
        operand->Info.Register.Reg = 0;
        break;

    case ND_OPT_MSR:
        // The operand is implicit and is a MSR (usually selected by the ECX register).
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_E;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = 0xFFFFFFFF;
        break;

    case ND_OPT_TSC:
        // The operand is implicit and is the IA32_TSC.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_TSC;
        break;

    case ND_OPT_TSCAUX:
        // The operand is implicit and is the IA32_TSCAUX.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_TSC_AUX;
        break;

    case ND_OPT_SCS:
        // The operand is implicit and is the IA32_SYSENTER_CS.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_SYSENTER_CS;
        break;

    case ND_OPT_SESP:
        // The operand is implicit and is the IA32_SYSENTER_ESP.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_SYSENTER_ESP;
        break;

    case ND_OPT_SEIP:
        // The operand is implicit and is the IA32_SYSENTER_EIP.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_SYSENTER_EIP;
        break;

    case ND_OPT_STAR:
        // The operand is implicit and is the IA32_STAR.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_STAR;
        break;

    case ND_OPT_LSTAR:
        // The operand is implicit and is the IA32_LSTAR.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_LSTAR;
        break;

    case ND_OPT_FMASK:
        // The operand is implicit and is the IA32_FMASK.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_FMASK;
        break;

    case ND_OPT_FSBASE:
        // The operand is implicit and is the IA32_FS_BASE MSR.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_FS_BASE;
        break;

    case ND_OPT_GSBASE:
        // The operand is implicit and is the IA32_GS_BASE MSR.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_GS_BASE;
        break;

    case ND_OPT_KGSBASE:
        // The operand is implicit and is the IA32_KERNEL_GS_BASE MSR.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_MSR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = NDR_IA32_KERNEL_GS_BASE;
        break;

    case ND_OPT_XCR:
        // The operand is implicit and is an extended control register (usually selected by ECX register).
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_E;
        operand->Info.Register.Type = ND_REG_XCR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = 0xFF;
        break;

    case ND_OPT_XCR0:
        // The operand is implicit and is XCR0.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_XCR;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = 0;
        break;

    case ND_OPT_BANK:
        // Multiple registers are accessed.
        if ((Instrux->Instruction == ND_INS_PUSHA) || (Instrux->Instruction == ND_INS_POPA))
        {
            operand->Type = ND_OP_REG;
            operand->Size = Instrux->WordLength;
            operand->Info.Register.Type = ND_REG_GPR;
            operand->Info.Register.Size = Instrux->WordLength;
            operand->Info.Register.Reg = NDR_EAX;
            operand->Info.Register.Count = 8;
            operand->Info.Register.IsBlock = ND_TRUE;
        }
        else
        {
            operand->Type = ND_OP_BANK;
        }
        break;

    case ND_OPT_A:
        // Fetch the address. NOTE: The size can't be larger than 8 bytes.
        if (ops == ND_OPS_p)
        {
            status = NdFetchAddressFar(Instrux, Code, Offset, Size, (ND_UINT8)size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }

            // Fill in operand info.
            operand->Type = ND_OP_ADDR_FAR;
            operand->Encoding = ND_OPE_D;
            operand->Info.Address.BaseSeg = Instrux->Address.Cs;
            operand->Info.Address.Offset = Instrux->Address.Ip;
        }
        else
        {
            status = NdFetchAddressNear(Instrux, Code, Offset, Size, (ND_UINT8)size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }

            // Fill in operand info.
            operand->Type = ND_OP_ADDR_NEAR;
            operand->Encoding = ND_OPE_D;
            operand->Info.AddressNear.Target = Instrux->AddressNear;
        }
        break;

    case ND_OPT_B:
        // General purpose register encoded in VEX.vvvv field.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_V;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)Instrux->Exs.v;

        // EVEX.V' must be 0, if a GPR is encoded using EVEX encoding.
        if (Instrux->Exs.vp != 0)
        {
            // If APX is present, V' can be used to extend the GPR to R16-R31.
            // Otherwise, #UD is triggered.
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Register.Reg |= Instrux->Exs.vp << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        break;

    case ND_OPT_C:
        // Control register, encoded in modrm.reg.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_CR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (Instrux->Exs.rp << 4) | (Instrux->Exs.r << 3) | Instrux->ModRm.reg;

        // On some AMD processors, the presence of the LOCK prefix before MOV to/from control registers allows accessing
        // higher 8 control registers.
        if ((ND_CODE_64 != Instrux->DefCode) && (Instrux->HasLock))
        {
            operand->Info.Register.Reg |= 0x8;
        }

        // Only CR0, CR2, CR3, CR4 & CR8 valid.
        if (operand->Info.Register.Reg != 0 &&
            operand->Info.Register.Reg != 2 &&
            operand->Info.Register.Reg != 3 &&
            operand->Info.Register.Reg != 4 &&
            operand->Info.Register.Reg != 8)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_D:
        // Debug register, encoded in modrm.reg.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_DR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (Instrux->Exs.rp << 4) | (Instrux->Exs.r << 3) | Instrux->ModRm.reg;

        // Only DR0-DR7 valid.
        if (operand->Info.Register.Reg >= 8)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_T:
        // Test register, encoded in modrm.reg.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_TR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.r << 3) | Instrux->ModRm.reg);

        // Only TR0-TR7 valid, only on 486.
        if (operand->Info.Register.Reg >= 8)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_S:
        // Segment register, encoded in modrm.reg.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_SEG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;

        // When addressing segment registers, any extension field (REX.R, REX2.R3, REX2.R4) is ignored.
        operand->Info.Register.Reg = Instrux->ModRm.reg;

        // Only ES, CS, SS, DS, FS, GS valid.
        if (operand->Info.Register.Reg >= 6)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        // If CS is loaded - #UD.
        if ((operand->Info.Register.Reg == NDR_CS) && operand->Access.Write)
        {
            return ND_STATUS_CS_LOAD;
        }

        break;

    case ND_OPT_E:
        // General purpose register or memory, encoded in modrm.rm.
        if (Instrux->ModRm.mod != 3)
        {
            goto memory;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)(Instrux->Exs.b << 3) | Instrux->ModRm.rm;

        // If APX is present, use B4 as well.
        if (Instrux->Exs.b4 != 0)
        {
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Register.Reg |= Instrux->Exs.b4 << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        operand->Info.Register.IsHigh8 = (operand->Info.Register.Size == 1) &&
                                         (operand->Info.Register.Reg  >= 4) &&
                                         (ND_ENCM_LEGACY == Instrux->EncMode) &&
                                         !Instrux->HasRex && !Instrux->HasRex2;
        break;

    case ND_OPT_F:
        // The flags register.
        operand->Type = ND_OP_REG;
        operand->Info.Register.Type = ND_REG_FLG;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = 0;
        Instrux->RflAccess |= operand->Access.Access;
        break;

    case ND_OPT_K:
        // The operand is the stack.
        {
            static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };

            Instrux->MemoryAccess |= operand->Access.Access;
            operand->Type = ND_OP_MEM;
            operand->Info.Memory.IsStack = ND_TRUE;
            operand->Info.Memory.HasBase = ND_TRUE;
            operand->Info.Memory.Base = NDR_RSP;
            operand->Info.Memory.BaseSize = szLut[Instrux->DefStack];
            operand->Info.Memory.HasSeg = ND_TRUE;
            operand->Info.Memory.Seg = NDR_SS;
            Instrux->StackWords = (ND_UINT8)(operand->Size / Instrux->WordLength);
            Instrux->StackAccess |= operand->Access.Access;
        }
        break;

    case ND_OPT_G:
        // General purpose register encoded in modrm.reg.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)(Instrux->Exs.r << 3) | Instrux->ModRm.reg;

        if (Instrux->Exs.rp != 0)
        {
            // If APX is present, use R' (R4) to extent the register to 5 bits.
            // Otherwise, generate #UD.
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Register.Reg |= Instrux->Exs.rp << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        operand->Info.Register.IsHigh8 = (operand->Info.Register.Size == 1) &&
                                         (operand->Info.Register.Reg  >= 4) &&
                                         (ND_ENCM_LEGACY == Instrux->EncMode) &&
                                         !Instrux->HasRex && !Instrux->HasRex2;
        break;

    case ND_OPT_R:
        // General purpose register encoded in modrm.rm.
        if ((Instrux->ModRm.mod != 3) && (0 == (Instrux->Attributes & ND_FLAG_MFR)))
        {
            return ND_STATUS_INVALID_ENCODING;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)(Instrux->Exs.b << 3) | Instrux->ModRm.rm;

        if (Instrux->Exs.b4 != 0)
        {
            // If APX is present, use B4 as well.
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Register.Reg |= Instrux->Exs.b4 << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        operand->Info.Register.IsHigh8 = (operand->Info.Register.Size == 1) &&
                                         (operand->Info.Register.Reg  >= 4) &&
                                         (ND_ENCM_LEGACY == Instrux->EncMode) &&
                                         !Instrux->HasRex && !Instrux->HasRex2;
        break;

    case ND_OPT_I:
        // Immediate, encoded in instructon bytes.
        {
            ND_UINT64 imm;

            // Fetch the immediate. NOTE: The size won't exceed 8 bytes.
            status = NdFetchImmediate(Instrux, Code, Offset, Size, (ND_UINT8)size);
            if (!ND_SUCCESS(status))
            {
                return status;
            }

            // Get the last immediate.
            if (Instrux->HasImm2)
            {
                imm = Instrux->Immediate2;
            }
            else
            {
                imm = Instrux->Immediate1;
            }

            operand->Type = ND_OP_IMM;
            operand->Encoding = ND_OPE_I;
            operand->Info.Immediate.RawSize = (ND_UINT8)size;

            if (operand->Flags.SignExtendedDws)
            {
                static const ND_UINT8 wszLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };

                // Get the default word size: the immediate is sign extended to the default word size.
                operand->Size = wszLut[Instrux->EfOpMode];

                operand->Info.Immediate.Imm = ND_SIGN_EX(size, imm);
            }
            else if (operand->Flags.SignExtendedOp1)
            {
                // The immediate is sign extended to the size of the first operand.
                operand->Size = Instrux->Operands[0].Size;

                operand->Info.Immediate.Imm = ND_SIGN_EX(size, imm);
            }
            else
            {
                operand->Info.Immediate.Imm = imm;
            }
        }
        break;

    case ND_OPT_m2zI:
        operand->Type = ND_OP_IMM;
        operand->Encoding = ND_OPE_L;
        operand->Info.Immediate.Imm = Instrux->SseImmediate & 3;
        operand->Info.Immediate.RawSize = (ND_UINT8)size;
        break;

    case ND_OPT_J:
        // Fetch the relative offset. NOTE: The size of the relative can't exceed 4 bytes.
        status = NdFetchRelativeOffset(Instrux, Code, Offset, Size, (ND_UINT8)size);
        if (!ND_SUCCESS(status))
        {
            return status;
        }

        // The instruction is RIP relative.
        Instrux->IsRipRelative = ND_TRUE;

        operand->Type = ND_OP_OFFS;
        operand->Encoding = ND_OPE_D;
        // The relative offset is forced to the default word length. Care must be taken with the 32 bit
        // branches that have 0x66 prefix (in 32 bit mode)!
        operand->Size = Instrux->WordLength;
        operand->Info.RelativeOffset.Rel = ND_SIGN_EX(size, Instrux->RelativeOffset);
        operand->Info.RelativeOffset.RawSize = (ND_UINT8)size;

        break;

    case ND_OPT_N:
        // The R/M field of the ModR/M byte selects a packed-quadword, MMX technology register.
        if (Instrux->ModRm.mod != 3)
        {
            return ND_STATUS_INVALID_ENCODING;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_MMX;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = Instrux->ModRm.rm;
        break;

    case ND_OPT_P:
        // The reg field of the ModR/M byte selects a packed quadword MMX technology register.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_MMX;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = Instrux->ModRm.reg;
        break;

    case ND_OPT_Q:
        // The rm field inside Mod R/M encodes a MMX register or memory.
        if (Instrux->ModRm.mod != 3)
        {
            goto memory;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_MMX;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = Instrux->ModRm.rm;
        break;

    case ND_OPT_O:
        // Absolute address, encoded in instruction bytes.
        // NOTE: The moffset len can't exceed 8 bytes.
        status = NdFetchMoffset(Instrux, Code, Offset, Size, 2 << Instrux->AddrMode);
        if (!ND_SUCCESS(status))
        {
            return status;
        }

        // operand info.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Encoding = ND_OPE_D;
        operand->Info.Memory.HasDisp = ND_TRUE;
        operand->Info.Memory.IsDirect = ND_TRUE;
        operand->Info.Memory.DispSize = Instrux->MoffsetLength;
        operand->Info.Memory.Disp = Instrux->Moffset;
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NdGetSegOverride(Instrux, NDR_DS);
        break;

    case ND_OPT_M:
        // Modrm based memory addressing.
        if (Instrux->ModRm.mod == 3)
        {
            return ND_STATUS_INVALID_ENCODING;
        }

memory:
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Encoding = ND_OPE_M;
        operand->Info.Memory.HasSeg = ND_TRUE;

        // Parse mode specific memory information.
        if (ND_ADDR_16 != Instrux->AddrMode)
        {
            status = NdParseMemoryOperand3264(Instrux, operand, vsibRegSize);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }
        else
        {
            status = NdParseMemoryOperand16(Instrux, operand);
            if (!ND_SUCCESS(status))
            {
                return status;
            }
        }

        // Get the segment. Note that in long mode, segment prefixes are ignored, except for FS and GS.
        if (Instrux->HasSeg)
        {
            operand->Info.Memory.Seg = NdGetSegOverride(Instrux, operand->Info.Memory.Seg);
        }

        // Handle VSIB addressing.
        if (ND_HAS_VSIB(Instrux))
        {
            // VSIB requires SIB.
            if (!Instrux->HasSib)
            {
                return ND_STATUS_VSIB_WITHOUT_SIB;
            }

            operand->Info.Memory.IsVsib = ND_TRUE;

            operand->Info.Memory.Vsib.IndexSize = vsibIndexSize;
            operand->Info.Memory.Vsib.ElemCount = vsibIndexCount;
            operand->Info.Memory.Vsib.ElemSize = (ND_UINT8)(size / vsibIndexCount);
        }

        // Handle sibmem addressing, as used by Intel AMX instructions.
        if (ND_HAS_SIBMEM(Instrux))
        {
            // sibmem requires SIB to be present.
            if (!Instrux->HasSib)
            {
                return ND_STATUS_SIBMEM_WITHOUT_SIB;
            }

            operand->Info.Memory.IsSibMem = ND_TRUE;
        }

        // If we have broadcast, the operand size is fixed to either 16, 32 or 64 bit, depending on bcast size.
        // Therefore, we will override the rawSize with either 16, 32 or 64 bits. Note that bcstSize will save the 
        // total size of the access, and it will be used to compute the number of broadcasted elements: 
        // bcstSize / rawSize.
        if (Instrux->HasBroadcast)
        {
            ND_OPERAND_SIZE bcstSize = size;
            operand->Info.Memory.HasBroadcast = ND_TRUE;

            if (opd & ND_OPD_B32)
            {
                size = ND_SIZE_32BIT;
            }
            else if (opd & ND_OPD_B64)
            {
                size = ND_SIZE_64BIT;
            }
            else if (opd & ND_OPD_B16)
            {
                size = ND_SIZE_16BIT;
            }
            else
            {
                size = width ? ND_SIZE_64BIT : ND_SIZE_32BIT;
            }

            // Override operand size.
            operand->Size = size;

            operand->Info.Memory.Broadcast.Size = (ND_UINT8)operand->Size;
            operand->Info.Memory.Broadcast.Count = (ND_UINT8)(bcstSize / operand->Size);
        }

        // Handle compressed displacement, if any. Note that most EVEX instructions with 8 bit displacement
        // use compressed displacement addressing.
        if (Instrux->HasCompDisp)
        {
            operand->Info.Memory.HasCompDisp = ND_TRUE;
            operand->Info.Memory.CompDispSize = NdGetCompDispSize(Instrux, operand->Size);
        }

        // MIB, if any. Used by some MPX instructions.
        operand->Info.Memory.IsMib = ND_HAS_MIB(Instrux);

        // Bitbase, if any. Used by BT* instructions when the first op is mem and the second one reg.
        operand->Info.Memory.IsBitbase = ND_HAS_BITBASE(Instrux);

        // AG, if this is the case.
        if (ND_HAS_AG(Instrux))
        {
            operand->Info.Memory.IsAG = ND_TRUE;

            // Address generation instructions ignore the segment prefixes. Examples are LEA and MPX instructions.
            operand->Info.Memory.HasSeg = ND_FALSE;
            operand->Info.Memory.Seg = 0;
        }

        // Shadow Stack Access, if this is the case.
        if (ND_HAS_SHS(Instrux))
        {
            operand->Info.Memory.IsShadowStack = ND_TRUE;
            operand->Info.Memory.ShStkType = ND_SHSTK_EXPLICIT;
        }

        break;


    case ND_OPT_H:
        // Vector register, encoded in VEX/EVEX.vvvv.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_V;
        operand->Info.Register.Type = ND_REG_SSE;
        operand->Info.Register.Size = (ND_REG_SIZE)(size < ND_SIZE_128BIT ? ND_SIZE_128BIT : size);
        // V' will be 0 for any non-EVEX encoded instruction.
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.vp << 4) | Instrux->Exs.v);
        break;

    case ND_OPT_L:
        // Vector register, encoded in immediate.
        status = NdFetchSseImmediate(Instrux, Code, Offset, Size, 1);
        if (!ND_SUCCESS(status))
        {
            return status;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_L;
        operand->Info.Register.Type = ND_REG_SSE;
        operand->Info.Register.Size = (ND_REG_SIZE)(size < ND_SIZE_128BIT ? ND_SIZE_128BIT : size);
        operand->Info.Register.Reg = (Instrux->SseImmediate >> 4) & 0xF;

        if (Instrux->DefCode != ND_CODE_64)
        {
            operand->Info.Register.Reg &= 0x7;
        }

        break;

    case ND_OPT_U:
        // Vector register encoded in modrm.rm.
        if (Instrux->ModRm.mod != 3)
        {
            return ND_STATUS_INVALID_ENCODING;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_SSE;
        operand->Info.Register.Size = (ND_REG_SIZE)(size < ND_SIZE_128BIT ? ND_SIZE_128BIT : size);
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.b << 3) | Instrux->ModRm.rm);

        if (Instrux->HasEvex)
        {
            operand->Info.Register.Reg |= Instrux->Exs.x << 4;
        }

        break;

    case ND_OPT_V:
        // Vector register encoded in modrm.reg.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_SSE;
        operand->Info.Register.Size = (ND_REG_SIZE)(size < ND_SIZE_128BIT ? ND_SIZE_128BIT : size);
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.r << 3) | Instrux->ModRm.reg);

        if (Instrux->HasEvex)
        {
            operand->Info.Register.Reg |= Instrux->Exs.rp << 4;
        }

        break;

    case ND_OPT_W:
        // Vector register or memory encoded in modrm.rm.
        if (Instrux->ModRm.mod != 3)
        {
            goto memory;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_SSE;
        operand->Info.Register.Size = (ND_REG_SIZE)(size < ND_SIZE_128BIT ? ND_SIZE_128BIT : size);
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.b << 3) | Instrux->ModRm.rm);

        // For vector registers, the X extension bit is used to extend the register to 5 bits.
        if (Instrux->HasEvex)
        {
            operand->Info.Register.Reg |= Instrux->Exs.x << 4;
        }

        break;

    case ND_OPT_X:
    case ND_OPT_Y:
    case ND_OPT_pDI:
        // RSI/RDI based addressing, as used by string instructions.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Base = (ND_UINT8)(((opt == ND_OPT_X) ? NDR_RSI : NDR_RDI));
        operand->Info.Memory.IsString = (ND_OPT_X == opt || ND_OPT_Y == opt);
        // DS:rSI supports segment overriding. ES:rDI does not.
        if (opt == ND_OPT_Y)
        {
            operand->Info.Memory.Seg = NDR_ES;
        }
        else
        {
            operand->Info.Memory.Seg = NdGetSegOverride(Instrux, NDR_DS);
        }
        break;

    case ND_OPT_pBXAL:
        // [rBX + AL], used by XLAT.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.HasIndex = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.IndexSize = ND_SIZE_8BIT;  // Always 1 Byte.
        operand->Info.Memory.Base = NDR_RBX;            // Always rBX.
        operand->Info.Memory.Index = NDR_AL;            // Always AL.
        operand->Info.Memory.Scale = 1;                 // Always 1.
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NdGetSegOverride(Instrux, NDR_DS);
        break;

    case ND_OPT_pAX:
        // [rAX], used implicitly by MONITOR, MONITORX and RMPADJUST instructions.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.Base = NDR_RAX;            // Always rAX.
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NdGetSegOverride(Instrux, NDR_DS);
        break;

    case ND_OPT_pCX:
        // [rCX], used implicitly by RMPUPDATE.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.Base = NDR_RCX;            // Always rCX.
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NdGetSegOverride(Instrux, NDR_DS);
        break;

    case ND_OPT_pBP:
        // [sBP], used implicitly by ENTER, when nesting level is > 1.
        // Operand size bytes accessed from memory. Base reg size determined by stack address size attribute.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->DefStack;
        operand->Info.Memory.Base = NDR_RBP;            // Always rBP.
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NDR_SS;
        break;

    case ND_OPT_SHS:
        // Shadow stack access using the current SSP.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.IsShadowStack = ND_TRUE;
        operand->Info.Memory.ShStkType = ND_SHSTK_SSP_LD_ST;
        break;

    case ND_OPT_SHS0:
        // Shadow stack access using the IA32_PL0_SSP.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.IsShadowStack = ND_TRUE;
        operand->Info.Memory.ShStkType = ND_SHSTK_PL0_SSP;
        break;

    case ND_OPT_SMT:
        // Table of MSR addresses, encoded in [RSI].
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.Base = NDR_RSI;            // Always rSI.
        operand->Info.Memory.HasSeg = ND_FALSE;         // Linear Address directly, only useable in 64 bit mode.
        break;

    case ND_OPT_DMT:
        // Table of MSR addresses, encoded in [RDI].
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.Base = NDR_RDI;            // Always rDI.
        operand->Info.Memory.HasSeg = ND_FALSE;         // Linear Address directly, only useable in 64 bit mode.
        break;

    case ND_OPT_SHSP:
        // Shadow stack push/pop access.
        Instrux->MemoryAccess |= operand->Access.Access;
        operand->Type = ND_OP_MEM;
        operand->Info.Memory.IsShadowStack = ND_TRUE;
        operand->Info.Memory.ShStkType = ND_SHSTK_SSP_PUSH_POP;
        break;

    case ND_OPT_Z:
        // A GPR Register is selected by the low 3 bits inside the opcode. REX.B can be used to extend it.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_O;
        operand->Info.Register.Type = ND_REG_GPR;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)(Instrux->Exs.b << 3) | (Instrux->PrimaryOpCode & 0x7);

        if (Instrux->Exs.b4 != 0)
        {
            // If APX is present, extend the register.
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Register.Reg |= Instrux->Exs.b4 << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        operand->Info.Register.IsHigh8 = (operand->Info.Register.Size == 1) &&
                                         (operand->Info.Register.Reg  >= 4) &&
                                         (ND_ENCM_LEGACY == Instrux->EncMode) &&
                                         !Instrux->HasRex && !Instrux->HasRex2;
        break;

    case ND_OPT_rB:
        // reg inside modrm selects a BND register.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_BND;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.r << 3) | Instrux->ModRm.reg);

        if (operand->Info.Register.Reg >= 4)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_mB:
        // rm inside modrm selects either a BND register, either memory.
        if (Instrux->ModRm.mod != 3)
        {
            goto memory;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_BND;
        operand->Info.Register.Size = (ND_REG_SIZE)size;
        operand->Info.Register.Reg = (ND_UINT8)((Instrux->Exs.b << 3) | Instrux->ModRm.rm);

        if (operand->Info.Register.Reg >= 4)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_rK:
        // reg inside modrm selects a mask register.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_MSK;

        // Opcode dependent #UD, R and R' must be zero (1 actually, but they're inverted).
        if ((Instrux->Exs.r != 0) || (Instrux->Exs.rp != 0))
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = (ND_UINT8)(Instrux->ModRm.reg);
        break;

    case ND_OPT_vK:
        // vex.vvvv selects a mask register.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_V;
        operand->Info.Register.Type = ND_REG_MSK;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = (ND_UINT8)Instrux->Exs.v;

        if (operand->Info.Register.Reg >= 8)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_mK:
        // rm inside modrm selects either a mask register, either memory.
        if (Instrux->ModRm.mod != 3)
        {
            goto memory;
        }

        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_MSK;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        // X and B are ignored when Msk registers are being addressed.
        operand->Info.Register.Reg = Instrux->ModRm.rm;
        break;

    case ND_OPT_aK:
        // aaa inside evex selects either a mask register, which is used for masking a destination operand.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_A;
        operand->Info.Register.Type = ND_REG_MSK;
        operand->Info.Register.Size = ND_SIZE_64BIT;
        operand->Info.Register.Reg = Instrux->Exs.k;
        break;

    case ND_OPT_rM:
        // Sigh. reg field inside mod r/m encodes memory. This encoding is used by MOVDIR64b and ENQCMD instructions.
        // When the ModR/M.reg field is used to select a memory operand, the following apply:
        // - The ES segment register is used as a base
        // - The ES segment register cannot be overridden
        // - The size of the base register is selected by the address size, not the operand size.
        operand->Type = ND_OP_MEM;
        operand->Encoding = ND_OPE_R;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.Base = (ND_UINT8)((Instrux->Exs.r << 3) | Instrux->ModRm.reg);

        if (Instrux->Exs.rp != 0)
        {
            // If APX is present, extend the base register.
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Memory.Base |= Instrux->Exs.rp << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NDR_ES;
        break;

    case ND_OPT_mM:
        // Sigh. rm field inside mod r/m encodes memory, even if mod is 3.
        operand->Type = ND_OP_MEM;
        operand->Encoding = ND_OPE_M;
        operand->Info.Memory.HasBase = ND_TRUE;
        operand->Info.Memory.Base = (ND_UINT8)((Instrux->Exs.b << 3) | Instrux->ModRm.rm);

        if (Instrux->Exs.b4 != 0)
        {
            // If APX is present, extend the base register.
            if (Instrux->FeatMode & ND_FEAT_APX)
            {
                operand->Info.Memory.Base |= Instrux->Exs.b4 << 4;
            }
            else
            {
                return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
            }
        }

        operand->Info.Memory.BaseSize = 2 << Instrux->AddrMode;
        operand->Info.Memory.HasSeg = ND_TRUE;
        operand->Info.Memory.Seg = NdGetSegOverride(Instrux, NDR_DS);
        break;

    case ND_OPT_rT:
        // Tile register encoded in ModR/M.reg field.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_R;
        operand->Info.Register.Type = ND_REG_TILE;
        operand->Info.Register.Size = size;
        operand->Info.Register.Reg = Instrux->ModRm.reg;

        // #UD if a tile register > 7 is encoded.
        if (Instrux->Exs.r != 0 || Instrux->Exs.rp != 0)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_mT:
        // Tile register encoded in ModR/M.rm field.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_M;
        operand->Info.Register.Type = ND_REG_TILE;
        operand->Info.Register.Size = size;
        operand->Info.Register.Reg = Instrux->ModRm.rm;

        // #UD if a tile register > 7 is encoded.
        if (Instrux->Exs.b != 0 || Instrux->Exs.b4 != 0)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_vT:
        // Tile register encoded in vex.vvvv field.
        operand->Type = ND_OP_REG;
        operand->Encoding = ND_OPE_V;
        operand->Info.Register.Type = ND_REG_TILE;
        operand->Info.Register.Size = size;
        operand->Info.Register.Reg = Instrux->Exs.v;

        // #UD if a tile register > 7 is encoded.
        if (operand->Info.Register.Reg > 7 || Instrux->Exs.vp != 0)
        {
            return ND_STATUS_INVALID_REGISTER_IN_INSTRUCTION;
        }

        break;

    case ND_OPT_dfv:
        // Default flags value encoded in vex.vvvv field.
        operand->Type = ND_OP_DFV;
        operand->Encoding = ND_OPE_V;
        operand->Info.DefaultFlags.CF = (Instrux->Exs.v >> 0) & 1;
        operand->Info.DefaultFlags.ZF = (Instrux->Exs.v >> 1) & 1;
        operand->Info.DefaultFlags.SF = (Instrux->Exs.v >> 2) & 1;
        operand->Info.DefaultFlags.OF = (Instrux->Exs.v >> 3) & 1;
        operand->Size = 0;
        break;

    default:
        return ND_STATUS_INVALID_INSTRUX;
    }

    if (operand->Type == ND_OP_REG)
    {
        // Handle block addressing - used by AVX512_4FMAPS and AVX512_4VNNIW instructions. Also used by VP2INTERSECTD/Q
        // instructions. Also note that in block addressing, the base of the block is masked using the size of the block;
        // for example, for a block size of 1, the first register must be even; For a block size of 4, the first register
        // must be divisible by 4.
        if (opb != 0)
        {
            operand->Info.Register.Count = opb;
            operand->Info.Register.Reg &= (ND_UINT32)~(opb - 1);
            operand->Info.Register.IsBlock = ND_TRUE;
        }
        else
        {
            operand->Info.Register.Count = 1;
        }

        // Handle zero-upper semantic for destination operands. Applies to destination registers only.
        if ((Instrux->HasNd || Instrux->HasZu) && operand->Access.Write && !operand->Flags.IsDefault)
        {
            operand->Info.Register.IsZeroUpper = 1;
        }
    }

    // Handle decorators. Note that only Mask, Zero and Broadcast are stored per-operand.
    if (0 != opd)
    {
        // Check for mask register. Mask if present only if the operand supports masking and if the
        // mask register is not k0 (which implies "no masking").
        if ((opd & ND_OPD_MASK) && (Instrux->HasMask))
        {
            operand->Decorator.HasMask = ND_TRUE;
            operand->Decorator.Msk = (ND_UINT8)Instrux->Exs.k;
        }

        // Check for zeroing. The operand must support zeroing and the z bit inside evex3 must be set. Note that
        // zeroing is allowed only for register destinations, and NOT for memory.
        if ((opd & ND_OPD_ZERO) && (Instrux->HasZero))
        {
            if (operand->Type == ND_OP_MEM)
            {
                return ND_STATUS_ZEROING_ON_MEMORY;
            }

            operand->Decorator.HasZero = ND_TRUE;
        }

        // Check for broadcast again. We've already filled the broadcast size before parsing the op size.
        if ((opd & ND_OPD_BCAST) && (Instrux->HasBroadcast))
        {
            operand->Decorator.HasBroadcast = ND_TRUE;
        }
    }

    return status;
}


//
// NdFindInstruction
//
static NDSTATUS
NdFindInstruction(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_SIZET Size,
    ND_IDBE **InsDef
    )
{
    NDSTATUS status;
    const ND_TABLE *pTable;
    ND_IDBE *pIns;
    ND_BOOL stop, redf2, redf3;
    ND_UINT32 nextOpcode;

    // pre-init
    status = ND_STATUS_SUCCESS;
    pIns = (ND_IDBE *)ND_NULL;
    stop = ND_FALSE;
    nextOpcode = 0;
    redf2 = redf3 = ND_FALSE;

    switch (Instrux->EncMode)
    {
    case ND_ENCM_LEGACY:
        if (Instrux->Rex2.m0 == 1)
        {
            // Legacy map ID 1.
            pTable = (const ND_TABLE*)gLegacyMap_opcode.Table[0x0F];
        }
        else
        {
            // Legacy map ID 0.
            pTable = (const ND_TABLE*)&gLegacyMap_opcode;
        }
        break;
    case ND_ENCM_XOP:
        pTable = (const ND_TABLE *)gXopMap_mmmmm.Table[Instrux->Exs.m];
        break;
    case ND_ENCM_VEX:
        pTable = (const ND_TABLE *)gVexMap_mmmmm.Table[Instrux->Exs.m];
        break;
    case ND_ENCM_EVEX:
        pTable = (const ND_TABLE *)gEvexMap_mmmmm.Table[Instrux->Exs.m];
        break;
    default:
        pTable = (const ND_TABLE *)ND_NULL;
        break;
    }

    while ((!stop) && (ND_NULL != pTable))
    {
        switch (pTable->Type)
        {
        case ND_ILUT_INSTRUCTION:
            // We've found the leaf entry, which is an instruction - we can leave.
            pIns = (ND_IDBE *)(((ND_TABLE_INSTRUCTION *)pTable)->Instruction);
            stop = ND_TRUE;
            break;

        case ND_ILUT_OPCODE:
            // We need an opcode to keep going.
            status = NdFetchOpcode(Instrux, Code, Instrux->Length, Size);
            if (!ND_SUCCESS(status))
            {
                stop = ND_TRUE;
                break;
            }

            pTable = (const ND_TABLE *)pTable->Table[Instrux->OpCodeBytes[nextOpcode++]];
            break;

        case ND_ILUT_OPCODE_LAST:
            // We need an opcode to select the next table, but the opcode is AFTER the modrm/sib/displacement.
            if (!Instrux->HasModRm)
            {
                // Fetch modrm, SIB & displacement
                status = NdFetchModrmSibDisplacement(Instrux, Code, Instrux->Length, Size);
                if (!ND_SUCCESS(status))
                {
                    stop = ND_TRUE;
                    break;
                }
            }

            // Fetch the opcode, which is after the modrm and displacement.
            status = NdFetchOpcode(Instrux, Code, Instrux->Length, Size);
            if (!ND_SUCCESS(status))
            {
                stop = ND_TRUE;
                break;
            }

            pTable = (const ND_TABLE *)pTable->Table[Instrux->OpCodeBytes[nextOpcode++]];
            break;

        case ND_ILUT_MODRM_MOD:
            // We need modrm.mod to select the next table.
            if (!Instrux->HasModRm)
            {
                // Fetch modrm, SIB & displacement
                status = NdFetchModrmSibDisplacement(Instrux, Code, Instrux->Length, Size);
                if (!ND_SUCCESS(status))
                {
                    stop = ND_TRUE;
                    break;
                }
            }

            // Next index is either 0 (mem) or 1 (reg)
            pTable = (const ND_TABLE *)pTable->Table[Instrux->ModRm.mod == 3 ? 1 : 0];
            break;

        case ND_ILUT_MODRM_REG:
            // We need modrm.reg to select the next table.
            if (!Instrux->HasModRm)
            {
                // Fetch modrm, SIB & displacement
                status = NdFetchModrmSibDisplacement(Instrux, Code, Instrux->Length, Size);
                if (!ND_SUCCESS(status))
                {
                    stop = ND_TRUE;
                    break;
                }
            }

            // Next index is the reg.
            pTable = (const ND_TABLE *)pTable->Table[Instrux->ModRm.reg];
            break;

        case ND_ILUT_MODRM_RM:
            // We need modrm.rm to select the next table.
            if (!Instrux->HasModRm)
            {
                // Fetch modrm, SIB & displacement
                status = NdFetchModrmSibDisplacement(Instrux, Code, Instrux->Length, Size);
                if (!ND_SUCCESS(status))
                {
                    stop = ND_TRUE;
                    break;
                }
            }

            // Next index is the rm.
            pTable = (const ND_TABLE *)pTable->Table[Instrux->ModRm.rm];
            break;

        case ND_ILUT_MAN_PREFIX:
            // We have mandatory prefixes.
            if ((Instrux->Rep == 0xF2) && !redf2)
            {
                // We can only redirect once through one mandatory prefix, otherwise we may
                // enter an infinite loop (see CRC32 Gw Eb -> 0x66 0xF2 0x0F ...)
                redf2 = ND_TRUE;
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_MAN_PREF_F2];
                Instrux->HasMandatoryF2 = ND_TRUE;
            }
            else if ((Instrux->Rep == 0xF3) && !redf3)
            {
                redf3 = ND_TRUE;
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_MAN_PREF_F3];
                Instrux->HasMandatoryF3 = ND_TRUE;
            }
            else if (Instrux->HasOpSize)
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_MAN_PREF_66];
                Instrux->HasMandatory66 = ND_TRUE;
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_MAN_PREF_NP];
            }
            break;

        case ND_ILUT_MODE:
            if (ND_NULL != pTable->Table[Instrux->DefCode + 1])
            {
                pTable = (const ND_TABLE *)pTable->Table[Instrux->DefCode + 1];
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_MODE_NONE];
            }
            break;

        case ND_ILUT_DSIZE:
            // Handle default/forced redirections in 64 bit mode.
            if (ND_CODE_64 == Instrux->DefCode)
            {
                // 64-bit mode, we may have forced/default operand sizes.
                if ((ND_NULL != pTable->Table[4]) && (!Instrux->HasOpSize || Instrux->Exs.w))
                {
                    pTable = (const ND_TABLE *)pTable->Table[4];
                }
                else if (ND_NULL != pTable->Table[5])
                {
                    pTable = (const ND_TABLE *)pTable->Table[5];
                }
                else if (ND_NULL != pTable->Table[Instrux->OpMode + 1])
                {
                    pTable = (const ND_TABLE *)pTable->Table[Instrux->OpMode + 1];
                }
                else
                {
                    pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_DSIZE_NONE];
                }
            }
            else if (ND_NULL != pTable->Table[Instrux->OpMode + 1])
            {
                pTable = (const ND_TABLE *)pTable->Table[Instrux->OpMode + 1];
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_DSIZE_NONE];
            }
            break;

        case ND_ILUT_ASIZE:
            if (ND_NULL != pTable->Table[Instrux->AddrMode + 1])
            {
                pTable = (const ND_TABLE *)pTable->Table[Instrux->AddrMode + 1];
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_ASIZE_NONE];
            }
            break;

        case ND_ILUT_AUXILIARY:
            // Auxiliary redirection. Default to table[0] if nothing matches.
            if ((Instrux->Exs.b || Instrux->Exs.b4) && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_REXB]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_REXB];
            }
            else if (Instrux->Exs.w && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_REXW]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_REXW];
            }
            else if ((Instrux->DefCode == ND_CODE_64) && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_MO64]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_MO64];
            }
            else if (Instrux->Rep == ND_PREFIX_G1_REPE_REPZ && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_REPZ]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_REPZ];
            }
            else if ((Instrux->Rep != 0) && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_REP]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_REP];
            }
            else if (Instrux->DefCode == ND_CODE_64 && Instrux->HasModRm && 
                Instrux->ModRm.mod == 0 && Instrux->ModRm.rm == NDR_RBP && 
                ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_RIPREL])
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_RIPREL];
            }
            else if (Instrux->HasRex2 && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_REX2]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_REX2];
            }
            else if (Instrux->HasRex2 && Instrux->Rex2.w && (ND_NULL != pTable->Table[ND_ILUT_INDEX_AUX_REX2W]))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_REX2W];
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_INDEX_AUX_NONE];
            }
            break;

        case ND_ILUT_VENDOR:
            // Vendor redirection. Go to the vendor specific entry.
            if (ND_NULL != pTable->Table[Instrux->VendMode])
            {
                pTable = (const ND_TABLE *)pTable->Table[Instrux->VendMode];
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_VEND_ANY];
            }
            break;

        case ND_ILUT_FEATURE:
            // Feature redirection. Normally NOP if feature is not set, but may be something else if feature is set.
            if ((ND_NULL != pTable->Table[ND_ILUT_FEATURE_MPX]) && !!(Instrux->FeatMode & ND_FEAT_MPX))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_FEATURE_MPX];
            }
            else if ((ND_NULL != pTable->Table[ND_ILUT_FEATURE_CET]) && !!(Instrux->FeatMode & ND_FEAT_CET))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_FEATURE_CET];
            }
            else if ((ND_NULL != pTable->Table[ND_ILUT_FEATURE_CLDEMOTE]) && !!(Instrux->FeatMode & ND_FEAT_CLDEMOTE))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_FEATURE_CLDEMOTE];
            }
            else if ((ND_NULL != pTable->Table[ND_ILUT_FEATURE_PITI]) && !!(Instrux->FeatMode & ND_FEAT_PITI))
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_FEATURE_PITI];
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[ND_ILUT_FEATURE_NONE];
            }
            break;

        case ND_ILUT_EX_M:
            pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.m];
            break;

        case ND_ILUT_EX_PP:
            pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.p];
            break;

        case ND_ILUT_EX_L:
            if (Instrux->HasEvex && Instrux->Exs.m != 4 && Instrux->Exs.bm)
            {
                // We have evex; we need to fetch the modrm now, because we have to make sure we don't have SAE or ER;
                // if we do have SAE or ER, we have to check the modrm byte and see if it is a reg-reg form (mod = 3),
                // in which case L'L is forced to the maximum vector length of the instruction. We know for sure that
                // all EVEX instructions have modrm.
                // Skip these checks for EVEX map 4, which are legacy instructions promoted to EVEX, and which do not
                // support SAE, ER or broadcast.
                if (!Instrux->HasModRm)
                {
                    // Fetch modrm, SIB & displacement
                    status = NdFetchModrmSibDisplacement(Instrux, Code, Instrux->Length, Size);
                    if (!ND_SUCCESS(status))
                    {
                        stop = ND_TRUE;
                        break;
                    }
                }

                if (3 == Instrux->ModRm.mod)
                {
                    // We use the maximum vector length of the instruction. If the instruction does not support
                    // SAE or ER, a #UD would be generated. We check for this later.
                    if (ND_NULL != pTable->Table[2])
                    {
                        pTable = (const ND_TABLE *)pTable->Table[2];
                    }
                    else if (ND_NULL != pTable->Table[1])
                    {
                        pTable = (const ND_TABLE *)pTable->Table[1];
                    }
                    else
                    {
                        pTable = (const ND_TABLE *)pTable->Table[0];
                    }
                }
                else
                {
                    // Mod is mem, we simply use L'L for indexing, as no SAE or ER can be present.
                    pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.l];
                }
            }
            else
            {
                pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.l];
            }
            break;

        case ND_ILUT_EX_W:
            pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.w];
            break;

        case ND_ILUT_EX_WI:
            pTable = (const ND_TABLE *)pTable->Table[Instrux->DefCode == ND_CODE_64 ? Instrux->Exs.w : 0];
            break;

        case ND_ILUT_EX_ND:
            // New data modified field encoded in EVEX payload byte 3.
            pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.nd];
            break;

        case ND_ILUT_EX_NF:
            // No flags modifier field encoded in EVEX payload byte 3.
            pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.nf];
            break;

        case ND_ILUT_EX_SC:
            // Standard condition field encoded in EVEX payload byte 3.
            pTable = (const ND_TABLE *)pTable->Table[Instrux->Exs.sc];
            break;

        default:
            status = ND_STATUS_INTERNAL_ERROR;
            stop = ND_TRUE;
            break;
        }
    }

    // Error - leave now.
    if (!ND_SUCCESS(status))
    {
        goto cleanup_and_exit;
    }

    // No encoding found - leave now.
    if (ND_NULL == pIns)
    {
        status = ND_STATUS_INVALID_ENCODING;
        goto cleanup_and_exit;
    }

    // Bingo! Valid instruction found for the encoding. If Modrm is needed and we didn't fetch it - do it now.
    if ((pIns->Attributes & ND_FLAG_MODRM) && (!Instrux->HasModRm))
    {
        if (0 == (pIns->Attributes & ND_FLAG_MFR))
        {
            // Fetch Mod R/M, SIB & displacement.
            status = NdFetchModrmSibDisplacement(Instrux, Code, Instrux->Length, Size);
            if (!ND_SUCCESS(status))
            {
                goto cleanup_and_exit;
            }
        }
        else
        {
            // Handle special MOV with control and debug registers - the mod is always forced to register. SIB
            // and displacement is ignored.
            status = NdFetchModrm(Instrux, Code, Instrux->Length, Size);
            if (!ND_SUCCESS(status))
            {
                goto cleanup_and_exit;
            }
        }
    }

    // Store primary opcode.
    Instrux->PrimaryOpCode = Instrux->OpCodeBytes[Instrux->OpLength - 1];

    Instrux->MainOpOffset = ND_IS_3DNOW(Instrux) ? Instrux->Length - 1 : Instrux->OpOffset + Instrux->OpLength - 1;

cleanup_and_exit:
    *InsDef = pIns;

    return status;
}


//
// NdGetAddrAndOpMode
//
static NDSTATUS
NdGetAddrAndOpMode(
    INSTRUX *Instrux
    )
{
    // Fill in addressing mode & default op size.
    switch (Instrux->DefCode)
    {
    case ND_CODE_16:
        Instrux->AddrMode = Instrux->HasAddrSize ? ND_ADDR_32 : ND_ADDR_16;
        Instrux->OpMode = Instrux->HasOpSize ? ND_OPSZ_32 : ND_OPSZ_16;
        break;
    case ND_CODE_32:
        Instrux->AddrMode = Instrux->HasAddrSize ? ND_ADDR_16 : ND_ADDR_32;
        Instrux->OpMode = Instrux->HasOpSize ? ND_OPSZ_16 : ND_OPSZ_32;
        break;
    case ND_CODE_64:
        Instrux->AddrMode = Instrux->HasAddrSize ? ND_ADDR_32 : ND_ADDR_64;
        Instrux->OpMode = Instrux->Exs.w ? ND_OPSZ_64 : (Instrux->HasOpSize ? ND_OPSZ_16 : ND_OPSZ_32);
        break;
    default:
        return ND_STATUS_INVALID_INSTRUX;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdGetEffectiveAddrAndOpMode
//
static NDSTATUS
NdGetEffectiveAddrAndOpMode(
    INSTRUX *Instrux
    )
{
    static const ND_UINT8 szLut[3] = { ND_SIZE_16BIT, ND_SIZE_32BIT, ND_SIZE_64BIT };
    ND_BOOL w64, f64, d64, has66;

    if ((ND_CODE_64 != Instrux->DefCode) && !!(Instrux->Attributes & ND_FLAG_IWO64))
    {
        // Some instructions ignore VEX/EVEX.W field outside 64 bit mode, and treat it as 0.
        Instrux->Exs.w = 0;
    }

    // Extract the flags.
    w64 = (0 != Instrux->Exs.w) && !(Instrux->Attributes & ND_FLAG_WIG);

    // In 64 bit mode, the operand is forced to 64 bit. Size-changing prefixes are ignored.
    f64 = 0 != (Instrux->Attributes & ND_FLAG_F64) && (ND_VEND_AMD != Instrux->VendMode);

    // In 64 bit mode, the operand defaults to 64 bit. No 32 bit form of the instruction exists. Note that on AMD,
    // only default 64 bit operands exist, even for branches - no operand is forced to 64 bit.
    d64 = (0 != (Instrux->Attributes & ND_FLAG_D64)) ||
          (0 != (Instrux->Attributes & ND_FLAG_F64) && (ND_VEND_AMD == Instrux->VendMode));

    // Check if 0x66 is indeed interpreted as a size changing prefix. Note that if 0x66 is a mandatory prefix,
    // then it won't be interpreted as a size changing prefix. However, there is an exception: MOVBE and CRC32
    // have mandatory 0xF2, and 0x66 is in fact a size changing prefix.
    // For legacy instructions promoted to EVEX, in some cases, the compressed prefix pp has the same meaning
    // as the legacy 0x66 prefix.
    has66 = (Instrux->HasOpSize && (!Instrux->HasMandatory66 || (Instrux->Attributes & ND_FLAG_S66))) || 
            ((Instrux->Exs.p == 1) && (Instrux->Attributes & ND_FLAG_SCALABLE));

    // Fill in the effective operand size. Also validate instruction validity in given mode.
    switch (Instrux->DefCode)
    {
    case ND_CODE_16:
        if (Instrux->Attributes & ND_FLAG_O64)
        {
            return ND_STATUS_INVALID_ENCODING_IN_MODE;
        }

        Instrux->EfOpMode = has66 ? ND_OPSZ_32 : ND_OPSZ_16;
        break;
    case ND_CODE_32:
        if (Instrux->Attributes & ND_FLAG_O64)
        {
            return ND_STATUS_INVALID_ENCODING_IN_MODE;
        }

        Instrux->EfOpMode = has66 ? ND_OPSZ_16 : ND_OPSZ_32;
        break;
    case ND_CODE_64:
        // Make sure instruction valid in mode.
        if (Instrux->Attributes & ND_FLAG_I64)
        {
            return ND_STATUS_INVALID_ENCODING_IN_MODE;
        }

        Instrux->EfOpMode = (w64 || f64 || (d64 && !has66)) ? ND_OPSZ_64 : (has66 ? ND_OPSZ_16 : ND_OPSZ_32);
        Instrux->AddrMode = !!(Instrux->Attributes & ND_FLAG_I67) ? ND_ADDR_64 : Instrux->AddrMode;
        break;
    default:
        return ND_STATUS_INVALID_INSTRUX;
    }

    // Fill in the default word length. It can't be more than 8 bytes.
    Instrux->WordLength = szLut[Instrux->EfOpMode];

    return ND_STATUS_SUCCESS;
}


//
// NdGetVectorLength
//
static NDSTATUS
NdGetVectorLength(
    INSTRUX *Instrux
    )
{
    if (Instrux->HasEr || Instrux->HasSae || Instrux->HasIgnEr)
    {
        // Embedded rounding or SAE present, force the vector length to 512 or scalar.
        if ((Instrux->TupleType == ND_TUPLE_T1S) || 
            (Instrux->TupleType == ND_TUPLE_T1S8) ||
            (Instrux->TupleType == ND_TUPLE_T1S16) ||
            (Instrux->TupleType == ND_TUPLE_T1F))
        {
            // Scalar instruction, vector length is 128 bits.
            Instrux->VecMode = Instrux->EfVecMode = ND_VECM_128;
        }
        else if (Instrux->Evex.u == 0)
        {
            // AVX 10 allows SAE/ER for 256-bit vector length, if EVEX.U is 0.
            // It is unclear whether the EVEX.U bit is ignored or reserved for scalar instructions.
            Instrux->VecMode = Instrux->EfVecMode = ND_VECM_256;
        }
        else
        {
            // Legacy or AVX 10 instruction with U bit set, vector length is 512 bits.
            Instrux->VecMode = Instrux->EfVecMode = ND_VECM_512;
        }

        return ND_STATUS_SUCCESS;
    }

    // Decode EVEX vector length. Also take into consideration the "ignore L" flag.
    switch (Instrux->Exs.l)
    {
    case 0:
        Instrux->VecMode = ND_VECM_128;
        Instrux->EfVecMode = ND_VECM_128;
        break;
    case 1:
        Instrux->VecMode = ND_VECM_256;
        Instrux->EfVecMode = (Instrux->Attributes & ND_FLAG_LIG) ? ND_VECM_128 : ND_VECM_256;
        break;
    case 2:
        Instrux->VecMode = ND_VECM_512;
        Instrux->EfVecMode = (Instrux->Attributes & ND_FLAG_LIG) ? ND_VECM_128 : ND_VECM_512;
        break;
    default:
        return ND_STATUS_BAD_EVEX_LL;
    }

    // Some instructions don't support 128 bit vectors.
    if ((ND_VECM_128 == Instrux->EfVecMode) && (0 != (Instrux->Attributes & ND_FLAG_NOL0)))
    {
        return ND_STATUS_INVALID_ENCODING;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdLegacyPrefixChecks
//
static NDSTATUS
NdLegacyPrefixChecks(
    INSTRUX *Instrux
    )
{
    // These checks only apply to legacy encoded instructions.

    // Check for LOCK. LOCK can be present only in two cases:
    // 1. For certain RMW instructions, as long as the destination operand is memory
    // 2. For MOV to/from CR0 in 32-bit mode on AMD CPUs, which allows access to CR8
    // For XOP/VEX/EVEX instructions, a #UD is generated (which is checked when fetching the XOP/VEX/EVEX prefix).
    if (Instrux->HasLock)
    {
        if (0 != (Instrux->Attributes & ND_FLAG_LOCK_SPECIAL) && (ND_CODE_32 == Instrux->DefCode))
        {
            // Special case of LOCK being used by MOV cr to access CR8.
        }
        else if (Instrux->ValidPrefixes.Lock && (Instrux->Operands[0].Type == ND_OP_MEM))
        {
            Instrux->IsLockEnabled = 1;
        }
        else
        {
            return ND_STATUS_BAD_LOCK_PREFIX;
        }
    }

    // Chec for REP prefixes. There are multiple uses:
    // 1. REP/REPNZ/REPZ, for string/IO instructions
    // 2. XACQUIRE/XRELEASE, for HLE-enabled instructions
    // 3. BND prefix, for branches
    // For XOP/VEX/EVEX instructions, a #UD is generated (which is checked when fetching the XOP/VEX/EVEX prefix).
    if (Instrux->Rep != 0)
    {
        if (Instrux->Attributes & ND_FLAG_NOREP)
        {
            return ND_STATUS_INVALID_ENCODING;
        }

        Instrux->IsRepEnabled = Instrux->ValidPrefixes.Rep != 0;

        Instrux->IsRepcEnabled = Instrux->ValidPrefixes.RepCond != 0;

        // Bound enablement.
        Instrux->IsBndEnabled = (Instrux->ValidPrefixes.Bnd != 0) && (Instrux->Rep == ND_PREFIX_G1_BND);

        // Check if the instruction is REPed.
        Instrux->IsRepeated = Instrux->IsRepEnabled || Instrux->IsRepcEnabled;

        // Check if the instruction is XACQUIRE or XRELEASE enabled.
        if ((Instrux->IsLockEnabled || Instrux->ValidPrefixes.HleNoLock) &&
            (Instrux->Operands[0].Type == ND_OP_MEM))
        {
            if ((Instrux->ValidPrefixes.Xacquire || Instrux->ValidPrefixes.Hle) && 
                (Instrux->Rep == ND_PREFIX_G1_XACQUIRE))
            {
                Instrux->IsXacquireEnabled = ND_TRUE;
            }
            else if ((Instrux->ValidPrefixes.Xrelease || Instrux->ValidPrefixes.Hle) && 
                (Instrux->Rep == ND_PREFIX_G1_XRELEASE))
            {
                Instrux->IsXreleaseEnabled = ND_TRUE;
            }
        }
    }

    // Check for segment prefixes. Besides offering segment override when accessing memory:
    // 1. Allow for branch hints to conditional branches
    // 2. Allow for Do Not Track prefix for indirect branches, to inhibit CET-IBT tracking
    // Segment prefixes are allowed with XOP/VEX/EVEX instructions, but they have the legacy meaning (no BHINT or DNT).
    if (Instrux->Seg != 0)
    {
        // Branch hint enablement.
        Instrux->IsBhintEnabled = Instrux->ValidPrefixes.Bhint && (
            (Instrux->Seg == ND_PREFIX_G2_BR_TAKEN) ||
            (Instrux->Seg == ND_PREFIX_G2_BR_NOT_TAKEN) ||
            (Instrux->Seg == ND_PREFIX_G2_BR_ALT));

        // Do-not-track hint enablement.
        Instrux->IsDntEnabled = Instrux->ValidPrefixes.Dnt && (Instrux->Seg == ND_PREFIX_G2_NO_TRACK);
    }

    // For XOP/VEX/EVEX instructions, a #UD is generated (which is checked when fetching the XOP/VEX/EVEX prefix).
    if (Instrux->HasOpSize && (Instrux->Attributes & ND_FLAG_NO66))
    {
        return ND_STATUS_INVALID_ENCODING;
    }

    // Address size override is allowed with all XOP/VEX/EVEX prefixes.
    if (Instrux->HasAddrSize && (Instrux->Attributes & ND_FLAG_NO67))
    {
        return ND_STATUS_INVALID_ENCODING;
    }

    // For XOP/VEX/EVEX instructions, a #UD is generated (which is checked when fetching the XOP/VEX/EVEX prefix).
    if (Instrux->HasRex2 && (Instrux->Attributes & ND_FLAG_NOREX2))
    {
        return ND_STATUS_INVALID_ENCODING;
    }

    // Check if the instruction is CET tracked. The do not track prefix (0x3E) works only for indirect near JMP and CALL
    // instructions. It is always enabled for far JMP and CALL instructions.
    Instrux->IsCetTracked = ND_HAS_CETT(Instrux) && !Instrux->IsDntEnabled;

    return ND_STATUS_SUCCESS;
}


//
// NdGetEvexFields
//
static NDSTATUS
NdGetEvexFields(
    INSTRUX *Instrux
    )
{
    // Validate the EVEX prefix, depending on the EVEX extension mode.
    if (Instrux->EvexMode == ND_EVEXM_EVEX)
    {
        // EVEX.U field must be 1 if the Modrm.Mod is not reg-reg OR if EVEX.b is 0.
        if (Instrux->Evex.u != 1 && (Instrux->ModRm.mod != 3 || Instrux->Exs.bm == 0))
        {
            return ND_STATUS_BAD_EVEX_U;
        }

        // Handle embedded broadcast/rounding-control.
        if (Instrux->Exs.bm == 1)
        {
            if (Instrux->ModRm.mod == 3)
            {
                // reg form for the instruction, check for ER or SAE support.
                if (Instrux->ValidDecorators.Er)
                {
                    Instrux->HasEr = 1;
                    Instrux->HasSae = 1;
                    Instrux->RoundingMode = (ND_UINT8)Instrux->Exs.l;
                }
                else if (Instrux->ValidDecorators.Sae)
                {
                    Instrux->HasSae = 1;
                }
                else if (!!(Instrux->Attributes & ND_FLAG_IER))
                {
                    // The encoding behaves as if embedded rounding is enabled, but it is in fact ignored.
                    Instrux->HasIgnEr = 1;
                }
                else
                {
                    return ND_STATUS_ER_SAE_NOT_SUPPORTED;
                }
            }
            else
            {
                // mem form for the instruction, check for broadcast.
                if (Instrux->ValidDecorators.Broadcast)
                {
                    Instrux->HasBroadcast = 1;
                }
                else
                {
                    return ND_STATUS_BROADCAST_NOT_SUPPORTED;
                }
            }
        }

        // Handle masking.
        if (Instrux->Exs.k != 0)
        {
            if (Instrux->ValidDecorators.Mask)
            {
                Instrux->HasMask = 1;
            }
            else
            {
                return ND_STATUS_MASK_NOT_SUPPORTED;
            }
        }
        else
        {
            if (!!(Instrux->Attributes & ND_FLAG_MMASK))
            {
                return ND_STATUS_MASK_REQUIRED;
            }
        }

        // Handle zeroing.
        if (Instrux->Exs.z != 0)
        {
            if (Instrux->ValidDecorators.Zero)
            {
                // Zeroing restrictions:
                // - valid with register only;
                // - valid only if masking is also used;
                if (Instrux->HasMask)
                {
                    Instrux->HasZero = 1;
                }
                else
                {
                    return ND_STATUS_ZEROING_NO_MASK;
                }
            }
            else
            {
                return ND_STATUS_ZEROING_NOT_SUPPORTED;
            }
        }

        // EVEX instructions with 8 bit displacement use compressed displacement addressing, where the displacement
        // is scaled according to the data type accessed by the instruction.
        if (Instrux->HasDisp && Instrux->DispLength == 1)
        {
            Instrux->HasCompDisp = ND_TRUE;
        }

        // Legacy EVEX.
        Instrux->Exs.nd = 0;
        Instrux->Exs.nf = 0;
        Instrux->Exs.sc = 0;
    }
    else
    {
        // EVEX extension for VEX/Legacy/Conditional instructions.
        const ND_UINT8 b3mask[4] =
        {         // Bit              7     6     5     4     3     2     1     0
            0x00, // Regular form: |  z  |  L  |  L  |  b  |  V4 |  a  |  a  |  a  |
            0xD3, // VEX form:     |  0  |  0  |  L  |  0  |  V4 |  NF |  0  |  0  |
            0xE3, // Legacy form:  |  0  |  0  |  0  |  ND |  V4 |  NF |  0  |  0  |
            0xE0, // Cond form:    |  0  |  0  |  0  |  ND | SC3 | SC2 | SC1 | SC0 |
        };

        // EVEX flavors are only valid in APX mode. Outside APX, only legacy EVEX is valid.
        if (0 == (Instrux->FeatMode & ND_FEAT_APX))
        {
            return ND_STATUS_INVALID_ENCODING;
        }

        // Apply EVEX payload byte 3 mask.
        if (0 != (Instrux->Evex.Evex[3] & b3mask[Instrux->EvexMode]))
        {
            return ND_STATUS_INVALID_EVEX_BYTE3;
        }

        // EVEX.U field must be 1 if mod is reg-reg.
        if (Instrux->Evex.u != 1 && Instrux->ModRm.mod == 3)
        {
            return ND_STATUS_BAD_EVEX_U;
        }

        if (Instrux->ValidDecorators.Nd)
        {
            Instrux->HasNd = (ND_BOOL)Instrux->Exs.nd;
        }

        if (Instrux->ValidDecorators.Nf)
        {
            Instrux->HasNf = (ND_BOOL)Instrux->Exs.nf;
        }

        if (Instrux->ValidDecorators.Zu)
        {
            Instrux->HasZu = (ND_BOOL)Instrux->Exs.nd;
        }

        Instrux->Exs.z = 0;
        Instrux->Exs.l = 0;
        Instrux->Exs.bm = 0;
        Instrux->Exs.k = 0;
    }

    return ND_STATUS_SUCCESS;
}


//
// NdVexExceptionChecks
//
static NDSTATUS
NdVexExceptionChecks(
    INSTRUX *Instrux
    )
{
    // These checks only apply to XOP/VEX/EVEX encoded instructions.

    // Instructions that don't use VEX/XOP/EVEX vvvv field must set it to 1111b/0 logic, otherwise a #UD will 
    // be generated.
    if ((Instrux->Attributes & ND_FLAG_NOV) && (0 != Instrux->Exs.v))
    {
        return ND_STATUS_VEX_VVVV_MUST_BE_ZERO;
    }

    // Instruction that don't use EVEX.V' field must set to to 1b/0 logic, otherwise a #UD will be generated.
    if ((Instrux->Attributes & ND_FLAG_NOVP) && (0 != Instrux->Exs.vp))
    {
        return ND_STATUS_BAD_EVEX_V_PRIME;
    }

    // VSIB instructions have a restriction: the same vector register can't be used by more than one operand.
    // The exception is SCATTER*, which can use the VSIB reg as two sources.
    if (ND_HAS_VSIB(Instrux) && Instrux->Category != ND_CAT_SCATTER)
    {
        ND_UINT8 usedVects[32] = { 0 };
        ND_UINT32 i;

        for (i = 0; i < Instrux->OperandsCount; i++)
        {
            if (Instrux->Operands[i].Type == ND_OP_REG && Instrux->Operands[i].Info.Register.Type == ND_REG_SSE)
            {
                if (++usedVects[Instrux->Operands[i].Info.Register.Reg] > 1)
                {
                    return ND_STATUS_INVALID_VSIB_REGS;
                }
            }
            else if (Instrux->Operands[i].Type == ND_OP_MEM)
            {
                if (++usedVects[Instrux->Operands[i].Info.Memory.Index] > 1)
                {
                    return ND_STATUS_INVALID_VSIB_REGS;
                }
            }
        }
    }

    // Handle AMX exception class.
    if (Instrux->ExceptionType == ND_EXT_AMX_E4 ||
        Instrux->ExceptionType == ND_EXT_AMX_E10)
    {
        // #UD if srcdest == src1, srcdest == src2 or src1 == src2. All three operands are tile regs.
        if (Instrux->Operands[0].Info.Register.Reg == Instrux->Operands[1].Info.Register.Reg ||
            Instrux->Operands[0].Info.Register.Reg == Instrux->Operands[2].Info.Register.Reg ||
            Instrux->Operands[1].Info.Register.Reg == Instrux->Operands[2].Info.Register.Reg)
        {
            return ND_STATUS_INVALID_TILE_REGS;
        }
    }

    // If E4* or E10* exception class is used (check out AVX512-FP16 instructions), an additional #UD case
    // exists: if the destination register is equal to either of the source registers.
    else if (Instrux->ExceptionType == ND_EXT_E4S || Instrux->ExceptionType == ND_EXT_E10S)
    {
        // Note that operand 0 is the destination, operand 1 is the mask, operand 2 is first source, operand
        // 3 is the second source.
        if (Instrux->Operands[0].Type == ND_OP_REG && Instrux->Operands[2].Type == ND_OP_REG &&
            Instrux->Operands[0].Info.Register.Reg == Instrux->Operands[2].Info.Register.Reg)
        {
            return ND_STATUS_INVALID_DEST_REGS;
        }

        if (Instrux->Operands[0].Type == ND_OP_REG && Instrux->Operands[3].Type == ND_OP_REG &&
            Instrux->Operands[0].Info.Register.Reg == Instrux->Operands[3].Info.Register.Reg)
        {
            return ND_STATUS_INVALID_DEST_REGS;
        }
    }

    // Handle PUSH2/POP2 exceptions, which have restrictions on the destination registers.
    else if (Instrux->ExceptionType == ND_EXT_APX_EVEX_PP2)
    {
        // The registers cannot be RSP for either PUSH2 or POP2.
        if (Instrux->Operands[0].Info.Register.Reg == NDR_RSP ||
            Instrux->Operands[1].Info.Register.Reg == NDR_RSP)
        {
            return ND_STATUS_INVALID_DEST_REGS;
        }

        // The destination registers cannot be the same for POP2.
        if (Instrux->Operands[0].Access.Write &&
            Instrux->Operands[0].Info.Register.Reg == Instrux->Operands[1].Info.Register.Reg)
        {
            return ND_STATUS_INVALID_DEST_REGS;
        }
    }

    return ND_STATUS_SUCCESS;
}


//
// NdCopyInstructionInfo
//
static NDSTATUS
NdCopyInstructionInfo(
    INSTRUX *Instrux,
    ND_IDBE *Idbe
    )
{
#ifndef BDDISASM_NO_MNEMONIC
    Instrux->Mnemonic = gMnemonics[Idbe->Mnemonic];
#endif // !BDDISASM_NO_MNEMONIC
    Instrux->Attributes = Idbe->Attributes;
    Instrux->Instruction = (ND_INS_CLASS)Idbe->Instruction;
    Instrux->Category = (ND_INS_CATEGORY)Idbe->Category;
    Instrux->IsaSet = (ND_INS_SET)Idbe->IsaSet;
    Instrux->FlagsAccess.Undefined.Raw = Idbe->SetFlags & Idbe->ClearedFlags;
    Instrux->FlagsAccess.Tested.Raw = Idbe->TestedFlags;
    Instrux->FlagsAccess.Modified.Raw = Idbe->ModifiedFlags;
    Instrux->FlagsAccess.Set.Raw = Idbe->SetFlags ^ Instrux->FlagsAccess.Undefined.Raw;
    Instrux->FlagsAccess.Cleared.Raw = Idbe->ClearedFlags ^ Instrux->FlagsAccess.Undefined.Raw;
    Instrux->CpuidFlag.Flag = Idbe->CpuidFlag;
    Instrux->ValidModes.Raw = Idbe->ValidModes;
    Instrux->ValidPrefixes.Raw = Idbe->ValidPrefixes;
    Instrux->ValidDecorators.Raw = Idbe->ValidDecorators;
    Instrux->FpuFlagsAccess.Raw = Idbe->FpuFlags;
    Instrux->SimdExceptions.Raw = Idbe->SimdExc;
    // Valid for EVEX, VEX and SSE instructions only. A value of 0 means it's not used.
    Instrux->ExceptionType = Idbe->ExcType;
    Instrux->TupleType = Idbe->TupleType;
    Instrux->EvexMode = Idbe->EvexMode;

    return ND_STATUS_SUCCESS;
}


//
// NdDecodeEx2
//
NDSTATUS
NdDecodeEx2(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_SIZET Size,
    ND_UINT8 DefCode,
    ND_UINT8 DefData,
    ND_UINT8 DefStack,
    ND_UINT8 Vendor
    )
{
    ND_CONTEXT opt;

    NdInitContext(&opt);

    opt.DefCode = DefCode;
    opt.DefData = DefData;
    opt.DefStack = DefStack;
    opt.VendMode = Vendor;
    opt.FeatMode = ND_FEAT_ALL; // Optimistically decode everything, as if all features are enabled.

    return NdDecodeWithContext(Instrux, Code, Size, &opt);
}


NDSTATUS
NdDecodeWithContext(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_SIZET Size,
    ND_CONTEXT *Context
    )
{
    NDSTATUS status;
    PND_IDBE pIns;
    ND_UINT32 opIndex;

    // pre-init
    status = ND_STATUS_SUCCESS;
    pIns = (PND_IDBE)ND_NULL;
    opIndex = 0;

    // validate
    if (ND_NULL == Instrux)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (ND_NULL == Code)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (Size == 0)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (ND_NULL == Context)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (ND_CODE_64 < Context->DefCode)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (ND_DATA_64 < Context->DefData)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (ND_VEND_MAX < Context->VendMode)
    {
        return ND_STATUS_INVALID_PARAMETER;
    }

    if (0 == (Context->Options & ND_OPTION_SKIP_ZERO_INSTRUX))
    {
        // Initialize with zero.
        nd_memzero(Instrux, sizeof(INSTRUX));
    }

    Instrux->DefCode = (ND_UINT8)Context->DefCode;
    Instrux->DefData = (ND_UINT8)Context->DefData;
    Instrux->DefStack = (ND_UINT8)Context->DefStack;
    Instrux->VendMode = (ND_UINT8)Context->VendMode;
    Instrux->FeatMode = (ND_UINT8)Context->FeatMode;
    Instrux->EncMode = ND_ENCM_LEGACY;  // Assume legacy encoding by default.

    // Fetch the instruction bytes.
    for (opIndex = 0; 
         opIndex < ((Size < ND_MAX_INSTRUCTION_LENGTH) ? Size : ND_MAX_INSTRUCTION_LENGTH); 
         opIndex++)
    {
        Instrux->InstructionBytes[opIndex] = Code[opIndex];
    }

    if (gPrefixesMap[Instrux->InstructionBytes[0]] != ND_PREF_CODE_NONE)
    {
        // Fetch prefixes. We peek at the first byte, to see if it's worth calling the prefix decoder.
        status = NdFetchPrefixes(Instrux, Instrux->InstructionBytes, 0, Size);
        if (!ND_SUCCESS(status))
        {
            return status;
        }
    }

    // Get addressing mode & operand size.
    status = NdGetAddrAndOpMode(Instrux);
    if (!ND_SUCCESS(status))
    {
        return status;
    }

    // Start iterating the tables, in order to extract the instruction entry.
    status = NdFindInstruction(Instrux, Instrux->InstructionBytes, Size, &pIns);
    if (!ND_SUCCESS(status))
    {
        return status;
    }

    // Copy information inside the Instrux.
    status = NdCopyInstructionInfo(Instrux, pIns);
    if (!ND_SUCCESS(status))
    {
        return status;
    }

    // Get effective operand mode.
    status = NdGetEffectiveAddrAndOpMode(Instrux);
    if (!ND_SUCCESS(status))
    {
        return status;
    }

    if (Instrux->HasEvex)
    {
        // Post-process EVEX encoded instructions. This does two thing:
        // - check and fill in decorator info;
        // - generate error for invalid broadcast/rounding, mask or zeroing bits;
        // - generate error if any reserved bits are set.
        status = NdGetEvexFields(Instrux);
        if (!ND_SUCCESS(status))
        {
            return status;
        }
    }

    if (ND_HAS_VECTOR(Instrux))
    {
        // Get vector length.
        status = NdGetVectorLength(Instrux);
        if (!ND_SUCCESS(status))
        {
            return status;
        }
    }

    Instrux->ExpOperandsCount = ND_EXP_OPS_CNT(pIns->OpsCount);
    Instrux->OperandsCount = Instrux->ExpOperandsCount;

    if (!(Context->Options & ND_OPTION_ONLY_EXPLICIT_OPERANDS))
    {
        Instrux->OperandsCount += ND_IMP_OPS_CNT(pIns->OpsCount);
    }

    // And now decode each operand.
    for (opIndex = 0; opIndex < Instrux->OperandsCount; ++opIndex)
    {
        status = NdParseOperand(Instrux, Instrux->InstructionBytes, Instrux->Length, Size, 
                                opIndex, pIns->Operands[opIndex]);
        if (!ND_SUCCESS(status))
        {
            return status;
        }
    }

    if (ND_ENCM_LEGACY == Instrux->EncMode)
    {
        // Do legacy prefix checks. Only available for legacy instructions. For XOP/VEX/EVEX instructions:
        // 1. LOCK, REP, 0x66, REX, REX2 cause #UD (checkd during XOP/VEX/EVEX fetch)
        // 2. Segment prefixes do not have BHINT or DNT semantic
        // 3. 0x67 can be used to override address mode
        // This has to be done after operand parsing, since some #UD conditions depend on them.
        status = NdLegacyPrefixChecks(Instrux);
        if (!ND_SUCCESS(status))
        {
            return status;
        }
    }
    else
    {
        // Do XOP/VEX/EVEX encoding checks. Additional #UD conditions, some dependent on encoded registers.
        // This has to be done after operand parsing, since some #UD conditions depend on them.
        status = NdVexExceptionChecks(Instrux);
        if (!ND_SUCCESS(status))
        {
            return status;
        }
    }

    // All done! Instruction successfully decoded!
    return ND_STATUS_SUCCESS;
}


//
// NdDecodeEx
//
NDSTATUS
NdDecodeEx(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_SIZET Size,
    ND_UINT8 DefCode,
    ND_UINT8 DefData
    )
{
    return NdDecodeEx2(Instrux, Code, Size, DefCode, DefData, DefCode, ND_VEND_ANY);
}


//
// NdDecode
//
NDSTATUS
NdDecode(
    INSTRUX *Instrux,
    const ND_UINT8 *Code,
    ND_UINT8 DefCode,
    ND_UINT8 DefData
    )
{
    return NdDecodeEx2(Instrux, Code, ND_MAX_INSTRUCTION_LENGTH, DefCode, DefData, DefCode, ND_VEND_ANY);
}


//
// NdInitContext
//
void
NdInitContext(
    ND_CONTEXT *Context
    )
{
    nd_memzero(Context, sizeof(*Context));
}
