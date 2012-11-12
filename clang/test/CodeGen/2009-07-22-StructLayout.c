// RUN: %clang_cc1 %s -triple i686-pc-linux-gnu -emit-llvm -o /dev/null
// PR4590

typedef unsigned char __u8;
typedef unsigned int __le32;
typedef unsigned int __u32;
typedef unsigned short __le16;
typedef unsigned short __u16;

struct usb_cdc_ether_desc {
 __u8 bLength;
 __u8 bDescriptorType;
 __u8 bDescriptorSubType;

 __u8 iMACAddress;
 __le32 bmEthernetStatistics;
 __le16 wMaxSegmentSize;
 __le16 wNumberMCFilters;
 __u8 bNumberPowerFilters;
} __attribute__ ((packed));


static struct usb_cdc_ether_desc ecm_desc __attribute__ ((__section__(".init.data"))) = {
 .bLength = sizeof ecm_desc,
 .bDescriptorType = ((0x01 << 5) | 0x04),
 .bDescriptorSubType = 0x0f,



 .bmEthernetStatistics = (( __le32)(__u32)(0)),
 .wMaxSegmentSize = (( __le16)(__u16)(1514)),
 .wNumberMCFilters = (( __le16)(__u16)(0)),
 .bNumberPowerFilters = 0,
};
